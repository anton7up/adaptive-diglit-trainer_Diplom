from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from app.services.error_analysis import build_error_analysis, prediction_error_frame
from app.services.feature_engineering import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from app.services.learning_curve import learning_curve_report, summarize_learning_curve


def frame(rows=180, users=9):
    rng = np.random.default_rng(11)
    user = np.arange(rows) % users
    difficulty = rng.uniform(0.1, 0.9, rows)
    ability = 0.2 + user / (users - 1) * 0.65
    target = (rng.random(rows) < 1 / (1 + np.exp(-5 * (ability - difficulty)))).astype(int)
    data = {name: rng.normal(size=rows) for name in NUMERIC_FEATURES}
    data["task_difficulty"] = difficulty
    data["skill_accuracy"] = np.clip(ability, 0, 1)
    data["user_accuracy_total"] = np.clip(ability, 0, 1)
    data["difficulty_gap"] = difficulty - ability
    data["skill_code"] = [f"S{i % 6}" for i in range(rows)]
    data["target"] = target
    data["user_id"] = user
    data["task_id"] = np.arange(rows) % 24
    data["attempt_id"] = np.arange(rows)
    data["created_at"] = pd.date_range("2026-01-01", periods=rows, freq="h")
    return pd.DataFrame(data)


def estimator():
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical = Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    return Pipeline([
        ("pre", ColumnTransformer([("num", numeric, NUMERIC_FEATURES), ("cat", categorical, CATEGORICAL_FEATURES)])),
        ("model", LogisticRegression(max_iter=1000)),
    ])


def test_error_frame_has_probability_columns():
    data = frame()
    model = estimator().fit(data[NUMERIC_FEATURES + CATEGORICAL_FEATURES], data["target"])
    errors = prediction_error_frame(model, data)
    assert "probability" in errors
    assert "absolute_error" in errors
    assert "error_type" in errors
    assert len(errors) == len(data)


def test_error_analysis_has_confidence_bands():
    data = frame()
    model = estimator().fit(data[NUMERIC_FEATURES + CATEGORICAL_FEATURES], data["target"])
    report = build_error_analysis(model, data)
    assert report["rows"] == len(data)
    assert report["confidence_bands"]
    assert report["hardest_skills"]


def test_learning_curve_has_multiple_points():
    data = frame(240, 12)
    model = estimator()
    rows = learning_curve_report(model, data, fractions=(0.4, 0.7, 1.0))
    assert len(rows) >= 2
    assert all("roc_auc" in row["metrics"] for row in rows)


def test_learning_curve_summary_has_conclusion():
    rows = [
        {"metrics": {"roc_auc": 0.60, "brier": 0.24}},
        {"metrics": {"roc_auc": 0.67, "brier": 0.20}},
    ]
    summary = summarize_learning_curve(rows)
    assert summary["conclusion"] == "additional_data_likely_useful"
