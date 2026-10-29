from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from .calibration import expected_calibration_error
from .feature_engineering import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET_COLUMN
from .model_training import probability_metrics
from .statistics_utils import bootstrap_confidence_interval, quantile_summary


def threshold_sweep(y_true: np.ndarray, probability: np.ndarray, thresholds: tuple[float, ...] | None = None) -> list[dict[str, float]]:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probability, dtype=float)
    thresholds = thresholds or tuple(float(x) for x in np.linspace(0.2, 0.8, 13))
    rows: list[dict[str, float]] = []
    for threshold in thresholds:
        predicted = (p >= threshold).astype(int)
        rows.append({
            "threshold": float(threshold),
            "accuracy": float(accuracy_score(y, predicted)),
            "precision": float(precision_score(y, predicted, zero_division=0)),
            "recall": float(recall_score(y, predicted, zero_division=0)),
            "f1": float(f1_score(y, predicted, zero_division=0)),
            "positive_prediction_rate": float(predicted.mean()),
        })
    return rows


def best_threshold(rows: list[dict[str, float]], metric: str = "f1") -> dict[str, float]:
    if not rows:
        raise ValueError("threshold rows are empty")
    if metric not in rows[0]:
        raise ValueError(f"unknown threshold metric: {metric}")
    return max(rows, key=lambda row: (float(row[metric]), -abs(float(row["threshold"]) - 0.5)))


def _bootstrap_metric_intervals(y: np.ndarray, p: np.ndarray, seed: int = 42) -> dict[str, dict[str, float]]:
    rng = np.random.default_rng(seed)
    indices = np.arange(len(y))
    auc_values: list[float] = []
    brier_values: list[float] = []
    logloss_values: list[float] = []
    for _ in range(250):
        sample = rng.choice(indices, size=len(indices), replace=True)
        ys = y[sample]
        ps = p[sample]
        if len(np.unique(ys)) < 2:
            continue
        metrics = probability_metrics(ys, ps)
        auc_values.append(metrics["roc_auc"])
        brier_values.append(metrics["brier"])
        logloss_values.append(metrics["log_loss"])
    return {
        "roc_auc": bootstrap_confidence_interval(np.asarray(auc_values), random_seed=seed).as_dict() if auc_values else {},
        "brier": bootstrap_confidence_interval(np.asarray(brier_values), random_seed=seed + 1).as_dict() if brier_values else {},
        "log_loss": bootstrap_confidence_interval(np.asarray(logloss_values), random_seed=seed + 2).as_dict() if logloss_values else {},
    }


def _group_metrics(frame: pd.DataFrame, probability: np.ndarray, column: str, minimum_rows: int = 20) -> list[dict[str, Any]]:
    if column not in frame:
        return []
    work = frame[[column, TARGET_COLUMN]].copy()
    work["probability"] = probability
    rows: list[dict[str, Any]] = []
    for value, group in work.groupby(column, dropna=False):
        if len(group) < minimum_rows:
            continue
        y = group[TARGET_COLUMN].astype(int).to_numpy()
        p = group["probability"].to_numpy(dtype=float)
        metrics = probability_metrics(y, p)
        rows.append({
            "group": str(value),
            "rows": int(len(group)),
            "positive_rate": float(y.mean()),
            "roc_auc": metrics["roc_auc"],
            "pr_auc": metrics["pr_auc"],
            "brier": metrics["brier"],
            "log_loss": metrics["log_loss"],
            "ece": expected_calibration_error(y, p, bins=6),
        })
    return sorted(rows, key=lambda row: (-row["rows"], row["group"]))


def _user_error_distribution(frame: pd.DataFrame, probability: np.ndarray) -> dict[str, Any]:
    if "user_id" not in frame:
        return {"users": 0, "brier": {}}
    work = frame[["user_id", TARGET_COLUMN]].copy()
    work["squared_error"] = (work[TARGET_COLUMN].astype(float) - probability) ** 2
    user_brier = work.groupby("user_id")["squared_error"].mean().to_numpy(dtype=float)
    return {
        "users": int(len(user_brier)),
        "brier": quantile_summary(user_brier) if len(user_brier) else {},
        "worst_decile_mean": float(np.mean(np.sort(user_brier)[-max(1, len(user_brier) // 10) :])) if len(user_brier) else None,
    }


def robustness_report(estimator, frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        raise ValueError("frame is empty")
    features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    y = frame[TARGET_COLUMN].astype(int).to_numpy()
    p = estimator.predict_proba(frame[features])[:, 1]
    thresholds = threshold_sweep(y, p)
    base = probability_metrics(y, p)
    by_skill = _group_metrics(frame, p, "skill_code", minimum_rows=15)
    by_task = _group_metrics(frame, p, "task_id", minimum_rows=12)
    return {
        "rows": int(len(frame)),
        "metrics": base,
        "bootstrap_intervals": _bootstrap_metric_intervals(y, p),
        "threshold_sweep": thresholds,
        "recommended_f1_threshold": best_threshold(thresholds, "f1"),
        "by_skill": by_skill,
        "by_task": by_task,
        "user_error_distribution": _user_error_distribution(frame, p),
        "diagnostics": {
            "skill_brier_range": (
                float(max(row["brier"] for row in by_skill) - min(row["brier"] for row in by_skill))
                if len(by_skill) >= 2 else 0.0
            ),
            "task_brier_range": (
                float(max(row["brier"] for row in by_task) - min(row["brier"] for row in by_task))
                if len(by_task) >= 2 else 0.0
            ),
            "note": "Subgroup differences are descriptive diagnostics and must not be interpreted as causal effects.",
        },
    }
