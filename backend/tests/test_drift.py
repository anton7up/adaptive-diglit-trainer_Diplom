from __future__ import annotations

import numpy as np
import pandas as pd

from app.services.drift_monitor import compare_feature_drift, drift_severity, population_stability_index
from app.services.feature_engineering import NUMERIC_FEATURES


def test_psi_is_near_zero_for_identical_distribution():
    values = np.linspace(0, 1, 1000)
    assert population_stability_index(values, values) < 1e-10


def test_psi_detects_shift():
    rng = np.random.default_rng(7)
    reference = rng.normal(0, 1, 2000)
    shifted = rng.normal(1.5, 1, 2000)
    assert population_stability_index(reference, shifted) > 0.25


def test_drift_severity_thresholds():
    assert drift_severity(0.01) == "stable"
    assert drift_severity(0.15) == "watch"
    assert drift_severity(0.30) == "drift"


def test_compare_feature_drift_sorts_descending():
    rng = np.random.default_rng(2)
    ref = pd.DataFrame({name: rng.normal(0, 1, 500) for name in NUMERIC_FEATURES})
    cur = ref.copy()
    cur[NUMERIC_FEATURES[0]] += 2.0
    signals = compare_feature_drift(ref, cur)
    assert signals[0].feature == NUMERIC_FEATURES[0]
    assert signals[0].psi >= signals[-1].psi
