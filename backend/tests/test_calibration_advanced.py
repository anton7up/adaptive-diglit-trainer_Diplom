from __future__ import annotations

import numpy as np
import pytest

from app.services.calibration import (
    brier_decomposition,
    calibration_report,
    calibration_slope_intercept,
    expected_calibration_error,
    maximum_calibration_error,
    reliability_bins,
)


def test_reliability_bins_preserve_sample_count():
    y = np.array([0, 0, 0, 1, 1, 1, 1, 0])
    p = np.array([0.05, 0.18, 0.31, 0.58, 0.67, 0.79, 0.92, 0.42])
    bins = reliability_bins(y, p, bins=5)
    assert sum(row.count for row in bins) == len(y)
    assert all(0 <= row.mean_probability <= 1 for row in bins)
    assert all(0 <= row.observed_rate <= 1 for row in bins)


def test_well_calibrated_predictions_have_lower_ece():
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1] * 30)
    good = np.array([0.10, 0.20, 0.25, 0.35, 0.65, 0.75, 0.80, 0.90] * 30)
    bad = np.where(y == 1, 0.25, 0.75)
    assert expected_calibration_error(y, good, bins=5) < expected_calibration_error(y, bad, bins=5)


def test_maximum_calibration_error_is_not_below_ece():
    y = np.array([0, 1, 0, 1, 0, 1])
    p = np.array([0.1, 0.8, 0.3, 0.6, 0.7, 0.4])
    assert maximum_calibration_error(y, p, bins=3) >= expected_calibration_error(y, p, bins=3)


def test_calibration_slope_for_informative_probabilities_is_positive():
    y = np.array([0] * 50 + [1] * 50)
    p = np.linspace(0.05, 0.95, 100)
    result = calibration_slope_intercept(y, p)
    assert result["slope"] > 0


def test_brier_decomposition_reconstructs_score_approximately():
    rng = np.random.default_rng(7)
    p = rng.uniform(0.05, 0.95, 400)
    y = (rng.random(400) < p).astype(int)
    result = brier_decomposition(y, p, bins=12)
    reconstructed = result["reliability"] - result["resolution"] + result["uncertainty"]
    assert abs(result["brier"] - reconstructed) < 0.03


def test_report_contains_diagnostic_fields():
    y = np.array([0, 0, 1, 1, 1, 0, 1, 0])
    p = np.array([0.1, 0.2, 0.7, 0.8, 0.6, 0.3, 0.9, 0.4])
    report = calibration_report(y, p, bins=4)
    assert report["samples"] == 8
    assert "ece" in report and "mce" in report
    assert "slope" in report and "intercept" in report
    assert "brier_decomposition" in report


@pytest.mark.parametrize(
    "y,p",
    [
        ([0, 1], [0.5]),
        ([], []),
        ([0, 2], [0.1, 0.9]),
        ([0, 1], [0.1, float("nan")]),
    ],
)
def test_invalid_calibration_inputs_are_rejected(y, p):
    with pytest.raises(ValueError):
        calibration_report(y, p)
