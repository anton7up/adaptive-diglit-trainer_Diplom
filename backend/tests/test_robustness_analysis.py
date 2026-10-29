from __future__ import annotations

import numpy as np
import pytest

from app.services.robustness_analysis import best_threshold, robustness_report, threshold_sweep
from ml_fixtures import fitted_estimator


def test_threshold_sweep_has_monotonic_thresholds():
    y = np.asarray([0, 0, 0, 1, 1, 1])
    p = np.asarray([0.05, 0.2, 0.45, 0.55, 0.8, 0.95])
    rows = threshold_sweep(y, p)
    thresholds = [row["threshold"] for row in rows]
    assert thresholds == sorted(thresholds)
    assert len(rows) == 13
    assert all(0.0 <= row["f1"] <= 1.0 for row in rows)


def test_threshold_sweep_higher_threshold_reduces_positive_rate():
    y = np.asarray([0, 0, 1, 1, 1])
    p = np.asarray([0.1, 0.35, 0.5, 0.7, 0.95])
    rows = threshold_sweep(y, p, thresholds=(0.3, 0.5, 0.8))
    rates = [row["positive_prediction_rate"] for row in rows]
    assert rates[0] >= rates[1] >= rates[2]


def test_best_threshold_selects_requested_metric():
    rows = [
        {"threshold": 0.4, "f1": 0.6, "accuracy": 0.8},
        {"threshold": 0.5, "f1": 0.8, "accuracy": 0.7},
        {"threshold": 0.6, "f1": 0.7, "accuracy": 0.9},
    ]
    assert best_threshold(rows, "f1")["threshold"] == 0.5
    assert best_threshold(rows, "accuracy")["threshold"] == 0.6


def test_best_threshold_rejects_unknown_metric():
    with pytest.raises(ValueError):
        best_threshold([{"threshold": 0.5, "f1": 0.7}], "missing")


def test_robustness_report_has_confidence_intervals():
    estimator, frame = fitted_estimator(300, 12)
    report = robustness_report(estimator, frame)
    assert report["rows"] == 300
    for metric in ("roc_auc", "brier", "log_loss"):
        interval = report["bootstrap_intervals"][metric]
        assert interval["lower"] <= interval["estimate"] <= interval["upper"]


def test_robustness_report_has_threshold_recommendation():
    estimator, frame = fitted_estimator(260, 13)
    report = robustness_report(estimator, frame)
    chosen = report["recommended_f1_threshold"]
    assert 0.2 <= chosen["threshold"] <= 0.8
    assert 0.0 <= chosen["f1"] <= 1.0


def test_robustness_report_segments_skills():
    estimator, frame = fitted_estimator(360, 12)
    report = robustness_report(estimator, frame)
    assert len(report["by_skill"]) == 6
    assert all(row["rows"] >= 15 for row in report["by_skill"])
    assert all(0.0 <= row["brier"] <= 1.0 for row in report["by_skill"])


def test_robustness_report_user_error_distribution():
    estimator, frame = fitted_estimator(360, 12)
    report = robustness_report(estimator, frame)
    distribution = report["user_error_distribution"]
    assert distribution["users"] == 12
    assert distribution["brier"]["count"] == 12
    assert distribution["worst_decile_mean"] >= distribution["brier"]["median"]


def test_robustness_report_rejects_empty_frame():
    estimator, frame = fitted_estimator(120, 8)
    with pytest.raises(ValueError):
        robustness_report(estimator, frame.iloc[0:0])
