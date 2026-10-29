from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from app.services.model_evaluation import calibration_bins, expected_calibration_error


def test_calibration_bins_cover_all_observations():
    y = np.array([0, 1, 0, 1, 1, 0, 1, 0])
    p = np.array([0.05, 0.95, 0.22, 0.72, 0.60, 0.40, 0.82, 0.18])
    rows = calibration_bins(y, p, bins=5)
    assert len(rows) == 5
    assert sum(row["count"] for row in rows) == len(y)


def test_ece_is_zero_for_perfect_bin_calibration():
    rows = [
        {"count": 50, "gap": 0.0},
        {"count": 50, "gap": 0.0},
    ]
    assert expected_calibration_error(rows) == 0.0


def test_ece_is_weighted_by_bin_size():
    rows = [
        {"count": 90, "gap": 0.1},
        {"count": 10, "gap": 0.5},
    ]
    assert abs(expected_calibration_error(rows) - 0.14) < 1e-9
