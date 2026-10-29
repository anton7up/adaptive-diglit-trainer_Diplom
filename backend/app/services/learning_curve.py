from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GroupShuffleSplit

from ..config import get_settings
from .feature_engineering import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET_COLUMN
from .model_training import probability_metrics


def learning_curve_report(
    estimator,
    frame: pd.DataFrame,
    fractions: tuple[float, ...] = (0.25, 0.40, 0.55, 0.70, 0.85, 1.0),
) -> list[dict[str, Any]]:
    features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    y = frame[TARGET_COLUMN].astype(int).to_numpy()
    groups = frame["user_id"].to_numpy()
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=get_settings().random_seed)
    train_idx, valid_idx = next(splitter.split(frame, y, groups))
    train = frame.iloc[train_idx].sort_values("created_at")
    valid = frame.iloc[valid_idx]
    y_valid = valid[TARGET_COLUMN].astype(int).to_numpy()
    output = []
    for fraction in fractions:
        count = max(20, int(len(train) * fraction))
        subset = train.iloc[:count]
        if subset[TARGET_COLUMN].nunique() < 2:
            continue
        model = clone(estimator)
        model.fit(subset[features], subset[TARGET_COLUMN].astype(int))
        probability = model.predict_proba(valid[features])[:, 1]
        metrics = probability_metrics(y_valid, probability)
        output.append({
            "fraction": fraction,
            "training_rows": len(subset),
            "validation_rows": len(valid),
            "metrics": metrics,
        })
    return output


def summarize_learning_curve(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"status": "insufficient_data", "rows": []}
    first = rows[0]["metrics"]
    last = rows[-1]["metrics"]
    auc_gain = last["roc_auc"] - first["roc_auc"]
    brier_gain = first["brier"] - last["brier"]
    if auc_gain > 0.03 or brier_gain > 0.02:
        conclusion = "additional_data_likely_useful"
    elif auc_gain < -0.02:
        conclusion = "performance_unstable"
    else:
        conclusion = "curve_near_plateau"
    return {
        "status": "ok",
        "conclusion": conclusion,
        "roc_auc_gain": float(auc_gain),
        "brier_improvement": float(brier_gain),
        "rows": rows,
    }
