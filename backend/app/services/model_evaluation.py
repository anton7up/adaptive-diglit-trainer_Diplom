from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import confusion_matrix

from ..config import get_settings
from ..timeutils import utcnow_naive
from .calibration import calibration_report
from .feature_engineering import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET_COLUMN
from .model_training import probability_metrics


def calibration_bins(y_true: np.ndarray, probability: np.ndarray, bins: int = 10) -> list[dict[str, float | int]]:
    edges = np.linspace(0.0, 1.0, bins + 1)
    output = []
    for index in range(bins):
        left, right = edges[index], edges[index + 1]
        mask = (probability >= left) & (probability < right if index < bins - 1 else probability <= right)
        count = int(mask.sum())
        if count == 0:
            output.append({"bin": index + 1, "from": float(left), "to": float(right), "count": 0, "mean_probability": 0.0, "observed_rate": 0.0, "gap": 0.0})
            continue
        mean_p = float(probability[mask].mean())
        observed = float(y_true[mask].mean())
        output.append({
            "bin": index + 1,
            "from": float(left),
            "to": float(right),
            "count": count,
            "mean_probability": mean_p,
            "observed_rate": observed,
            "gap": abs(mean_p - observed),
        })
    return output


def expected_calibration_error(rows: list[dict[str, float | int]]) -> float:
    total = sum(int(row["count"]) for row in rows)
    if total == 0:
        return 0.0
    return float(sum(int(row["count"]) / total * float(row["gap"]) for row in rows))


def evaluate_estimator(estimator, frame: pd.DataFrame) -> dict[str, Any]:
    features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    x = frame[features]
    y = frame[TARGET_COLUMN].astype(int).to_numpy()
    probability = estimator.predict_proba(x)[:, 1]
    prediction = (probability >= 0.5).astype(int)
    metrics = probability_metrics(y, probability)
    calibration = calibration_report(y, probability, bins=10)
    matrix = confusion_matrix(y, prediction, labels=[0, 1]).tolist()
    return {
        "metrics": metrics,
        "calibration": calibration,
        "calibration_bins": [
            {
                "bin": index + 1,
                "from": row["lower"],
                "to": row["upper"],
                "count": row["count"],
                "mean_probability": row["mean_probability"],
                "observed_rate": row["observed_rate"],
                "gap": row["absolute_gap"],
            }
            for index, row in enumerate(calibration["bins"])
        ],
        "expected_calibration_error": calibration["ece"],
        "confusion_matrix": matrix,
    }


def segment_metrics(estimator, frame: pd.DataFrame, column: str) -> list[dict[str, Any]]:
    if column not in frame.columns:
        raise ValueError(column)
    rows = []
    for value, subset in frame.groupby(column):
        if len(subset) < 5:
            continue
        result = evaluate_estimator(estimator, subset)
        rows.append({"segment": str(value), "rows": len(subset), **result["metrics"]})
    return rows


def permutation_feature_importance(estimator, frame: pd.DataFrame, max_rows: int = 1500) -> list[dict[str, float | str]]:
    features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    sample = frame.tail(max_rows)
    x = sample[features]
    y = sample[TARGET_COLUMN].astype(int)
    if y.nunique() < 2:
        return []
    result = permutation_importance(
        estimator,
        x,
        y,
        scoring="neg_brier_score",
        n_repeats=5,
        random_state=get_settings().random_seed,
    )
    rows = [
        {
            "feature": feature,
            "importance_mean": float(result.importances_mean[index]),
            "importance_std": float(result.importances_std[index]),
        }
        for index, feature in enumerate(features)
    ]
    return sorted(rows, key=lambda item: abs(float(item["importance_mean"])), reverse=True)


def build_evaluation_report(estimator, frame: pd.DataFrame, model_version: str) -> dict[str, Any]:
    report = evaluate_estimator(estimator, frame)
    report.update({
        "model_version": model_version,
        "generated_at": utcnow_naive().isoformat() + "Z",
        "rows": int(len(frame)),
        "positive_rate": float(frame[TARGET_COLUMN].mean()),
        "by_skill": segment_metrics(estimator, frame, "skill_code"),
        "feature_importance": permutation_feature_importance(estimator, frame),
    })
    return report


def save_evaluation_report(report: dict[str, Any]) -> Path:
    settings = get_settings()
    path = settings.report_dir / f"evaluation-{report['model_version']}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
