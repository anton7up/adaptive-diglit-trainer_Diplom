from __future__ import annotations

import numpy as np
import pytest

from app.services.statistics_utils import (
    bootstrap_confidence_interval,
    bootstrap_difference_interval,
    gini_coefficient,
    normalized_entropy,
    quantile_summary,
    rank_percentile,
    safe_divide,
    weighted_mean,
)


def test_bootstrap_interval_contains_mean_for_regular_sample():
    values = np.linspace(0.1, 0.9, 50)
    result = bootstrap_confidence_interval(values, iterations=300)
    assert result.lower <= result.estimate <= result.upper
    assert result.samples == 50


def test_paired_difference_detects_consistent_advantage():
    left = np.full(40, 0.10)
    right = np.full(40, 0.20)
    result = bootstrap_difference_interval(left, right, iterations=200)
    assert result.upper < 0


def test_quantile_summary_is_ordered():
    result = quantile_summary([1, 2, 3, 4, 5])
    assert result["min"] <= result["p25"] <= result["median"] <= result["p75"] <= result["max"]


def test_normalized_entropy_extremes():
    assert normalized_entropy([100, 0, 0]) == 0.0
    assert normalized_entropy([10, 10, 10]) == pytest.approx(1.0)


def test_gini_extremes():
    assert gini_coefficient([1, 1, 1, 1]) == pytest.approx(0.0)
    assert gini_coefficient([0, 0, 0, 10]) > 0.70


def test_safe_divide_and_weighted_mean():
    assert safe_divide(1, 0, default=-1) == -1
    assert weighted_mean([1, 3], [1, 1]) == pytest.approx(2.0)


def test_weighted_mean_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        weighted_mean([1, 2], [1])


def test_rank_percentile():
    assert rank_percentile([1, 2, 3, 4], 2) == pytest.approx(0.5)


def test_paired_randomization_detects_consistent_direction():
    from app.services.statistics_utils import paired_randomization_pvalue, paired_standardized_effect

    slower = np.asarray([12, 14, 11, 15, 13, 16, 12, 14], dtype=float)
    faster = np.asarray([9, 10, 9, 11, 10, 12, 8, 11], dtype=float)
    p_value = paired_randomization_pvalue(slower, faster, alternative="greater")
    assert p_value < 0.05
    assert paired_standardized_effect(slower, faster) > 1.0


def test_holm_adjustment_is_monotone_and_bounded():
    from app.services.statistics_utils import holm_adjusted_pvalues

    raw = [0.01, 0.04, 0.03, 0.20]
    adjusted = holm_adjusted_pvalues(raw)
    assert len(adjusted) == len(raw)
    assert all(0.0 <= value <= 1.0 for value in adjusted)
    assert adjusted[0] == pytest.approx(0.04)
    assert adjusted[3] == pytest.approx(0.20)
