#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from sqlalchemy.orm import joinedload

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import models
from app.config import get_settings
from app.database import SessionLocal
from app.services.ablation import run_ablation
from app.services.classical_baselines import benchmark_classical_baselines
from app.services.cohort_analysis import attach_group_labels, evaluate_cohorts
from app.services.data_quality import inspect_training_frame
from app.services.error_analysis import build_error_analysis
from app.services.explainability import global_sensitivity
from app.services.feature_engineering import build_training_frame
from app.services.learning_curve import learning_curve_report, summarize_learning_curve
from app.services.model_card import build_model_card
from app.services.model_evaluation import build_evaluation_report
from app.services.model_registry import ModelRegistry
from app.services.monitoring_service import model_monitoring_report
from app.services.policy_simulation import run_policy_robustness, run_policy_simulation
from app.services.retraining_policy import evaluate_retraining_need
from app.services.temporal_backtest import expanding_window_backtest
from app.timeutils import utcnow_naive


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the complete ML research report bundle for the diploma project.")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--simulation-learners", type=int, default=60)
    parser.add_argument("--simulation-steps", type=int, default=90)
    parser.add_argument("--robustness-learners", type=int, default=24)
    parser.add_argument("--mastery-threshold", type=float, default=0.70)
    parser.add_argument("--minimum-skill-mastery", type=float, default=0.50)
    parser.add_argument("--skip-simulation", action="store_true")
    parser.add_argument("--skip-policy-robustness", action="store_true")
    parser.add_argument("--skip-ablation", action="store_true")
    parser.add_argument("--skip-backtest", action="store_true")
    return parser.parse_args()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def main() -> int:
    args = parse_args()
    settings = get_settings()
    output_dir = args.output_dir or settings.report_dir / f"research-suite-{utcnow_naive().strftime('%Y%m%d-%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)

    db = SessionLocal()
    try:
        registry = ModelRegistry(db)
        record, estimator = registry.load_active()
        if record is None or estimator is None:
            print("No active model. Run scripts/bootstrap_and_train.py first.", file=sys.stderr)
            return 2

        attempts = (
            db.query(models.Attempt)
            .options(joinedload(models.Attempt.task).joinedload(models.Task.skill))
            .order_by(models.Attempt.created_at.asc(), models.Attempt.id.asc())
            .all()
        )
        frame = build_training_frame(attempts)
        if frame.empty:
            print("No attempts are available.", file=sys.stderr)
            return 3

        quality = inspect_training_frame(frame).as_dict()
        metadata = registry.metadata(record.version)
        evaluation = build_evaluation_report(estimator, frame, record.version)
        model_card = build_model_card(record, metadata, quality)
        errors = build_error_analysis(estimator, frame)
        curve = summarize_learning_curve(learning_curve_report(estimator, frame))
        sensitivity = global_sensitivity(estimator, frame)
        cohorts = evaluate_cohorts(estimator, attach_group_labels(db, frame), min_rows=20, min_users=2)
        monitoring = model_monitoring_report(db)
        retraining = evaluate_retraining_need(db).as_dict()

        reports = {
            "quality": quality,
            "evaluation": evaluation,
            "model_card": model_card,
            "error_analysis": errors,
            "learning_curve": curve,
            "global_sensitivity": sensitivity,
            "cohort_analysis": cohorts,
            "monitoring": monitoring,
            "retraining_policy": retraining,
            "classical_baselines": benchmark_classical_baselines(frame, metadata),
        }

        if not args.skip_ablation:
            reports["ablation"] = run_ablation(frame)
        if not args.skip_backtest:
            try:
                reports["temporal_backtest"] = expanding_window_backtest(estimator, frame, windows=4)
            except ValueError as exc:
                reports["temporal_backtest"] = {"status": "not_available", "reason": str(exc)}
        if not args.skip_simulation:
            tasks = (
                db.query(models.Task)
                .options(joinedload(models.Task.skill))
                .filter(models.Task.active.is_(True))
                .all()
            )
            reports["policy_simulation"] = run_policy_simulation(
                tasks,
                estimator,
                learners=args.simulation_learners,
                steps=args.simulation_steps,
                random_seed=settings.random_seed,
                mastery_threshold=args.mastery_threshold,
                minimum_skill_mastery=args.minimum_skill_mastery,
            )
            if not args.skip_policy_robustness:
                reports["policy_robustness"] = run_policy_robustness(
                    tasks,
                    estimator,
                    learners=args.robustness_learners,
                    steps=args.simulation_steps,
                    mastery_threshold=args.mastery_threshold,
                    minimum_skill_mastery=args.minimum_skill_mastery,
                )

        defense_summary = {
            "data_status": "synthetic bootstrap only; no human efficacy claim",
            "primary_endpoint": reports.get("policy_simulation", {}).get("primary_endpoint"),
            "learning_gain_endpoint": reports.get("policy_simulation", {}).get("learning_gain_endpoint"),
            "policy_robustness": reports.get("policy_robustness", {}).get("summary"),
            "classical_baselines": {
                "best_by_brier": reports["classical_baselines"].get("best_by_brier"),
                "models": reports["classical_baselines"].get("models"),
                "ml_vs_classical": reports["classical_baselines"].get("ml_vs_classical"),
            },
            "novelty": [
                "ML probability estimation is separated from the pedagogical task-selection policy.",
                "Adaptivity is evaluated by time-to-mastery and mastery gain rather than predictive AUC alone.",
                "The policy result is repeated across five independent simulation seeds to assess Monte Carlo stability.",
                "The active ML predictor is compared with Elo, online 2PL IRT, BKT and PFA on the same held-out learner split.",
                "All policy simulations are reproducible and use matched initial learner profiles and random streams.",
            ],
            "human_validation_status": "not conducted; randomized human pilot remains required",
        }
        reports["defense_summary"] = defense_summary

        files = {}
        for name, report in reports.items():
            path = output_dir / f"{name}.json"
            write_json(path, report)
            files[name] = path.name

        index = {
            "generated_at": utcnow_naive().isoformat() + "Z",
            "model_version": record.version,
            "model_name": record.model_name,
            "dataset_rows": int(len(frame)),
            "dataset_users": int(frame["user_id"].nunique()),
            "feature_count": len(record.feature_names),
            "files": files,
        }
        write_json(output_dir / "index.json", index)
        print(json.dumps({"output_dir": str(output_dir), **index}, ensure_ascii=False, indent=2))
        return 0
    finally:
        db.close()


if __name__ == "__main__":
    raise SystemExit(main())
