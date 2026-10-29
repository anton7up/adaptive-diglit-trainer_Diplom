from __future__ import annotations

import numpy as np

from app.services.dataset_profile import (
    build_dataset_profile,
    categorical_feature_profile,
    exposure_profile,
    numeric_feature_profile,
    suspicious_feature_correlations,
    temporal_profile,
)
from app.services.feature_engineering import NUMERIC_FEATURES
from ml_fixtures import make_frame


def test_dataset_profile_reports_expected_shape():
    frame = make_frame(300, 12)
    report = build_dataset_profile(frame)
    assert report["rows"] == 300
    assert report["features"] >= 35
    assert report["numeric_features"] == len(NUMERIC_FEATURES)
    assert report["categorical_features"] == 1
    assert report["target"]["positive"] + report["target"]["negative"] == 300
    assert 0.0 < report["target"]["positive_rate"] < 1.0


def test_numeric_profile_covers_every_numeric_feature():
    frame = make_frame(180, 9)
    rows = numeric_feature_profile(frame)
    assert len(rows) == len(NUMERIC_FEATURES)
    assert all(row["present"] for row in rows)
    assert all(row["rows"] == 180 for row in rows)
    assert all(0.0 <= row["missing_rate"] <= 1.0 for row in rows)


def test_numeric_profile_detects_missing_values():
    frame = make_frame(100, 8)
    frame.loc[:9, "response_time_last_sec"] = np.nan
    row = next(item for item in numeric_feature_profile(frame) if item["feature"] == "response_time_last_sec")
    assert row["missing"] == 10
    assert row["missing_rate"] == 0.1


def test_numeric_profile_calculates_target_correlation():
    frame = make_frame(240, 12)
    rows = numeric_feature_profile(frame)
    difficulty = next(item for item in rows if item["feature"] == "task_difficulty")
    assert difficulty["target_correlation"] is not None
    assert -1.0 <= difficulty["target_correlation"] <= 1.0


def test_categorical_profile_has_entropy_and_shares():
    frame = make_frame(240, 12)
    rows = categorical_feature_profile(frame)
    assert len(rows) == 1
    skill = rows[0]
    assert skill["feature"] == "skill_code"
    assert skill["unique"] == 6
    assert 0.0 <= skill["entropy"] <= 1.0
    assert abs(sum(item["share"] for item in skill["top_values"]) - 1.0) < 1e-9


def test_exposure_profile_summarizes_users_and_tasks():
    frame = make_frame(240, 12)
    users = exposure_profile(frame, "user_id")
    tasks = exposure_profile(frame, "task_id")
    assert users["entities"] == 12
    assert tasks["entities"] == 36
    assert users["minimum"] > 0
    assert 0.0 <= users["gini"] <= 1.0
    assert 0.0 < tasks["top_share"] < 1.0


def test_temporal_profile_orders_bounds():
    frame = make_frame(120, 8)
    report = temporal_profile(frame)
    assert report["start"] < report["end"]
    assert report["days"] > 0
    assert report["rows_per_day"] > 0


def test_suspicious_feature_correlations_detects_duplicate_feature():
    frame = make_frame(160, 8)
    frame["user_accuracy_last10"] = frame["user_accuracy_total"]
    rows = suspicious_feature_correlations(frame, threshold=0.999)
    pairs = {(row["left"], row["right"]) for row in rows}
    assert ("user_accuracy_total", "user_accuracy_last10") in pairs or ("user_accuracy_last10", "user_accuracy_total") in pairs


def test_dataset_profile_lists_strongest_univariate_correlations():
    report = build_dataset_profile(make_frame(320, 16))
    strongest = report["strongest_univariate_correlations"]
    assert 1 <= len(strongest) <= 12
    assert all("feature" in row and "target_correlation" in row for row in strongest)
