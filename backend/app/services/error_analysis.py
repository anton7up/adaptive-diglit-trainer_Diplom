from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd

from .feature_engineering import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET_COLUMN


def prediction_error_frame(estimator, frame: pd.DataFrame) -> pd.DataFrame:
    features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    output = frame.copy()
    probability = estimator.predict_proba(frame[features])[:, 1]
    output["probability"] = probability
    output["prediction"] = (probability >= 0.5).astype(int)
    output["absolute_error"] = np.abs(probability - output[TARGET_COLUMN].astype(float))
    output["squared_error"] = (probability - output[TARGET_COLUMN].astype(float)) ** 2
    output["error_type"] = np.select(
        [
            (output["prediction"] == 1) & (output[TARGET_COLUMN] == 0),
            (output["prediction"] == 0) & (output[TARGET_COLUMN] == 1),
        ],
        ["false_positive", "false_negative"],
        default="correct_classification",
    )
    return output


def confidence_error_bands(errors: pd.DataFrame) -> list[dict[str, Any]]:
    boundaries = [(0.0, 0.55), (0.55, 0.70), (0.70, 0.85), (0.85, 1.01)]
    output = []
    confidence = np.maximum(errors["probability"], 1.0 - errors["probability"])
    for lower, upper in boundaries:
        mask = (confidence >= lower) & (confidence < upper)
        subset = errors.loc[mask]
        if len(subset) == 0:
            continue
        output.append({
            "confidence_from": lower,
            "confidence_to": min(upper, 1.0),
            "rows": len(subset),
            "error_rate": float((subset["prediction"] != subset[TARGET_COLUMN]).mean()),
            "mean_absolute_probability_error": float(subset["absolute_error"].mean()),
        })
    return output


def hardest_segments(errors: pd.DataFrame, min_rows: int = 20) -> list[dict[str, Any]]:
    rows = []
    for skill_code, subset in errors.groupby("skill_code"):
        if len(subset) < min_rows:
            continue
        rows.append({
            "segment": str(skill_code),
            "rows": len(subset),
            "mean_absolute_error": float(subset["absolute_error"].mean()),
            "brier": float(subset["squared_error"].mean()),
            "false_positive_rate": float((subset["error_type"] == "false_positive").mean()),
            "false_negative_rate": float((subset["error_type"] == "false_negative").mean()),
        })
    return sorted(rows, key=lambda item: item["brier"], reverse=True)


def top_high_confidence_errors(errors: pd.DataFrame, limit: int = 25) -> list[dict[str, Any]]:
    wrong = errors[errors["prediction"] != errors[TARGET_COLUMN]].copy()
    wrong["confidence"] = np.maximum(wrong["probability"], 1.0 - wrong["probability"])
    wrong = wrong.sort_values(["confidence", "absolute_error"], ascending=False).head(limit)
    columns = ["attempt_id", "user_id", "task_id", "skill_code", "target", "probability", "prediction", "confidence", "error_type"]
    return [
        {key: _native(value) for key, value in row.items()}
        for row in wrong[columns].to_dict(orient="records")
    ]


def _native(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def build_error_analysis(estimator, frame: pd.DataFrame) -> dict[str, Any]:
    errors = prediction_error_frame(estimator, frame)
    return {
        "rows": len(errors),
        "classification_error_rate": float((errors["prediction"] != errors[TARGET_COLUMN]).mean()),
        "mean_absolute_probability_error": float(errors["absolute_error"].mean()),
        "confidence_bands": confidence_error_bands(errors),
        "hardest_skills": hardest_segments(errors),
        "high_confidence_errors": top_high_confidence_errors(errors),
    }
