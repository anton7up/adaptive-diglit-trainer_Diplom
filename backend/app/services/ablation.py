from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ..config import get_settings
from .feature_engineering import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET_COLUMN

FEATURE_GROUPS = {
    "long_term_history": [
        "user_attempts_total", "user_accuracy_total", "skill_attempts", "skill_accuracy",
        "task_attempts", "task_accuracy", "task_exposure_share", "skill_exposure_share",
    ],
    "recent_performance": [
        "user_accuracy_last5", "user_accuracy_last10", "skill_accuracy_last5", "skill_accuracy_last10",
        "correct_streak", "incorrect_streak", "skill_correct_streak", "skill_incorrect_streak",
    ],
    "practice_frequency": [
        "user_attempts_24h", "user_attempts_7d", "skill_attempts_7d", "practice_density_7d",
        "seconds_since_last_attempt", "seconds_since_last_skill_attempt", "seconds_since_last_task_attempt",
    ],
    "timing": [
        "response_time_mean_sec", "response_time_std_sec", "response_time_last_sec",
        "skill_response_time_mean_sec", "skill_response_time_last_sec", "speed_ratio",
        "task_estimated_time_sec", "local_hour_sin", "local_hour_cos",
    ],
    "task_characteristics": [
        "task_difficulty", "task_discrimination", "task_cognitive_level", "difficulty_gap",
        "is_first_skill_attempt", "is_first_task_attempt",
    ],
    "categorical": ["skill_code"],
}



def _pipeline(numeric: list[str], categorical: list[str]) -> Pipeline:
    transformers = []
    if numeric:
        transformers.append(("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), numeric))
    if categorical:
        transformers.append(("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), categorical))
    return Pipeline([
        ("preprocess", ColumnTransformer(transformers)),
        ("model", LogisticRegression(max_iter=1200, class_weight="balanced", random_state=get_settings().random_seed)),
    ])


def _score(frame: pd.DataFrame, features: list[str]) -> dict[str, float]:
    numeric = [item for item in features if item in NUMERIC_FEATURES]
    categorical = [item for item in features if item in CATEGORICAL_FEATURES]
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=get_settings().random_seed)
    y = frame[TARGET_COLUMN].astype(int).to_numpy()
    groups = frame["user_id"].to_numpy()
    train_idx, valid_idx = next(splitter.split(frame, y, groups))
    model = _pipeline(numeric, categorical)
    model.fit(frame.iloc[train_idx][features], y[train_idx])
    p = model.predict_proba(frame.iloc[valid_idx][features])[:, 1]
    return {
        "roc_auc": float(roc_auc_score(y[valid_idx], p)),
        "brier": float(brier_score_loss(y[valid_idx], p)),
    }


def run_ablation(frame: pd.DataFrame) -> dict[str, Any]:
    all_features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    baseline = _score(frame, all_features)
    rows = []
    for group_name, removed in FEATURE_GROUPS.items():
        remaining = [feature for feature in all_features if feature not in removed]
        score = _score(frame, remaining)
        rows.append({
            "removed_group": group_name,
            "removed_features": removed,
            "roc_auc": score["roc_auc"],
            "brier": score["brier"],
            "delta_roc_auc": score["roc_auc"] - baseline["roc_auc"],
            "delta_brier": score["brier"] - baseline["brier"],
        })
    return {"baseline": baseline, "ablations": rows}
