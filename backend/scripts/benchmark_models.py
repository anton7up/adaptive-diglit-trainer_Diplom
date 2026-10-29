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
from app.services.experiment_analysis import repeated_group_benchmark
from app.services.feature_engineering import build_training_frame
from app.services.model_experiments import experiment_specs
from app.services.model_training import candidate_estimators
from app.timeutils import utcnow_naive


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a reproducible group-aware benchmark of model families.")
    parser.add_argument("--full", action="store_true", help="Benchmark all hyperparameter variants instead of one representative per family.")
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def representative_specs(seed: int):
    specs = experiment_specs(seed)
    result = []
    seen = set()
    for spec in specs:
        if spec.family not in seen:
            result.append(spec)
            seen.add(spec.family)
    return result


def main() -> int:
    args = parse_args()
    settings = get_settings()
    db = SessionLocal()
    try:
        attempts = (
            db.query(models.Attempt)
            .options(joinedload(models.Attempt.task).joinedload(models.Task.skill))
            .order_by(models.Attempt.created_at.asc(), models.Attempt.id.asc())
            .all()
        )
        frame = build_training_frame(attempts)
        specs = experiment_specs(settings.random_seed) if args.full else representative_specs(settings.random_seed)
        estimators = candidate_estimators(settings.random_seed, specs)
        report = repeated_group_benchmark(
            frame,
            estimators,
            folds=args.folds,
            repetitions=args.repetitions,
            random_seed=settings.random_seed,
        )
        output = args.output or settings.report_dir / f"benchmark-{utcnow_naive().strftime('%Y%m%d-%H%M%S')}.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({
            "output": str(output),
            "models": report["models"],
            "rows": report["rows"],
            "users": report["users"],
            "winner": report["summary"][0]["model"],
        }, ensure_ascii=False, indent=2))
        return 0
    finally:
        db.close()


if __name__ == "__main__":
    raise SystemExit(main())
