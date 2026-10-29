from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np


METRIC_DIRECTIONS = {
    "accuracy": "max",
    "precision": "max",
    "recall": "max",
    "f1": "max",
    "roc_auc": "max",
    "pr_auc": "max",
    "log_loss": "min",
    "brier": "min",
}


def build_selection_report(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    if not candidates:
        return {"candidates": 0, "ranking": [], "metric_winners": {}, "stability": {}}
    ranking = []
    metric_winners: dict[str, dict[str, Any]] = {}
    for metric, direction in METRIC_DIRECTIONS.items():
        available = [row for row in candidates if metric in row.get("validation", {})]
        if not available:
            continue
        chooser = max if direction == "max" else min
        winner = chooser(available, key=lambda row: float(row["validation"][metric]))
        metric_winners[metric] = {
            "model": winner["name"],
            "value": float(winner["validation"][metric]),
            "direction": direction,
        }

    families: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        family = _family(row["name"])
        families[family].append(row)
        cv_rows = row.get("cross_validation", [])
        cv_scores = [_composite_like(item) for item in cv_rows if item]
        ranking.append({
            "name": row["name"],
            "family": family,
            "composite_score": float(row.get("composite_score", 0.0)),
            "validation": row.get("validation", {}),
            "cv_fold_count": len(cv_scores),
            "cv_composite_mean": float(np.mean(cv_scores)) if cv_scores else None,
            "cv_composite_std": float(np.std(cv_scores)) if cv_scores else None,
        })
    ranking.sort(key=lambda item: item["composite_score"], reverse=True)

    family_summary = []
    for family, rows in families.items():
        scores = [float(row.get("composite_score", 0.0)) for row in rows]
        family_summary.append({
            "family": family,
            "variants": len(rows),
            "best_score": max(scores),
            "mean_score": float(np.mean(scores)),
            "std_score": float(np.std(scores)),
            "best_variant": max(rows, key=lambda row: float(row.get("composite_score", 0.0)))["name"],
        })
    family_summary.sort(key=lambda item: item["best_score"], reverse=True)
    return {
        "candidates": len(candidates),
        "ranking": ranking,
        "metric_winners": metric_winners,
        "family_summary": family_summary,
        "stability": _stability(ranking),
        "selection_rationale": _selection_rationale(ranking, metric_winners),
    }


def _family(name: str) -> str:
    if name.startswith("logreg"):
        return "logistic_regression"
    if name.startswith("rf_"):
        return "random_forest"
    if name.startswith("extra_"):
        return "extra_trees"
    if name.startswith("gb_"):
        return "gradient_boosting"
    return name.split("_")[0]


def _composite_like(metrics: dict[str, float]) -> float:
    roc = float(metrics.get("roc_auc", 0.5))
    pr = float(metrics.get("pr_auc", 0.5))
    brier = float(metrics.get("brier", 0.25))
    f1 = float(metrics.get("f1", 0.5))
    log_loss = float(metrics.get("log_loss", 0.693))
    return 0.30 * roc + 0.20 * pr + 0.20 * max(0.0, 1.0 - brier) + 0.15 * f1 + 0.15 / (1.0 + log_loss)


def _stability(ranking: list[dict[str, Any]]) -> dict[str, Any]:
    if not ranking:
        return {}
    best = ranking[0]
    second = ranking[1] if len(ranking) > 1 else None
    margin = best["composite_score"] - second["composite_score"] if second else best["composite_score"]
    unstable = best["cv_composite_std"] is not None and best["cv_composite_std"] > 0.06
    return {
        "winner": best["name"],
        "margin_to_second": float(margin),
        "winner_cv_std": best["cv_composite_std"],
        "winner_is_unstable": bool(unstable),
        "close_race": bool(second is not None and margin < 0.015),
    }


def _selection_rationale(ranking: list[dict[str, Any]], metric_winners: dict[str, dict[str, Any]]) -> list[str]:
    if not ranking:
        return []
    winner = ranking[0]
    wins = [metric for metric, row in metric_winners.items() if row["model"] == winner["name"]]
    rationale = [
        f"{winner['name']} has the highest composite model-selection score ({winner['composite_score']:.4f}).",
    ]
    if wins:
        rationale.append(f"It also leads the holdout comparison on: {', '.join(wins)}.")
    if winner["cv_composite_std"] is not None:
        rationale.append(f"Its cross-validation composite standard deviation is {winner['cv_composite_std']:.4f}.")
    rationale.append("The final choice prioritizes probability quality and calibration, not classification accuracy alone.")
    return rationale
