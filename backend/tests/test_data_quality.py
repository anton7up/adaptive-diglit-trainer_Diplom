from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.services.data_quality import assert_training_quality, inspect_training_frame
from app.services.feature_engineering import CATEGORICAL_FEATURES, NUMERIC_FEATURES


def make_frame(rows=300, users=10, tasks=20):
    rng = np.random.default_rng(4)
    data = {feature: rng.normal(size=rows) for feature in NUMERIC_FEATURES}
    data["skill_code"] = np.array([f"S{i % 6}" for i in range(rows)])
    data["target"] = np.array([i % 2 for i in range(rows)])
    data["user_id"] = np.array([i % users for i in range(rows)])
    data["task_id"] = np.array([i % tasks for i in range(rows)])
    data["attempt_id"] = np.arange(rows)
    data["created_at"] = pd.date_range("2026-01-01", periods=rows, freq="h")
    return pd.DataFrame(data)


def test_quality_report_passes_reasonable_frame():
    report = inspect_training_frame(make_frame())
    assert report.passed
    assert report.rows == 300
    assert report.users == 10
    assert report.tasks == 20


def test_too_few_rows_fails_gate():
    report = inspect_training_frame(make_frame(rows=100))
    check = next(item for item in report.checks if item.name == "row_count")
    assert check.status == "fail"


def test_too_few_users_fails_gate():
    report = inspect_training_frame(make_frame(users=2))
    check = next(item for item in report.checks if item.name == "user_count")
    assert check.status == "fail"


def test_duplicate_attempt_ids_fail_gate():
    frame = make_frame()
    frame.loc[1, "attempt_id"] = frame.loc[0, "attempt_id"]
    report = inspect_training_frame(frame)
    check = next(item for item in report.checks if item.name == "duplicate_attempt_ids")
    assert check.status == "fail"


def test_missing_feature_fails_schema_gate():
    frame = make_frame().drop(columns=[NUMERIC_FEATURES[0]])
    report = inspect_training_frame(frame)
    check = next(item for item in report.checks if item.name == "feature_schema")
    assert check.status == "fail"


def test_non_finite_value_fails_gate():
    frame = make_frame()
    frame.loc[0, NUMERIC_FEATURES[0]] = np.inf
    report = inspect_training_frame(frame)
    check = next(item for item in report.checks if item.name == "finite_numeric")
    assert check.status == "fail"


def test_assert_quality_raises_on_failure():
    with pytest.raises(ValueError, match="quality gate failed"):
        assert_training_quality(make_frame(rows=80, users=2, tasks=5))
