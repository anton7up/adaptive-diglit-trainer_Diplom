from __future__ import annotations

from dataclasses import dataclass
from math import log
from typing import Callable, Iterable

import numpy as np


@dataclass(frozen=True)
class ConfidenceInterval:
    estimate: float
    lower: float
    upper: float
    confidence: float
    samples: int

    def as_dict(self) -> dict[str, float | int]:
        return {
            "estimate": self.estimate,
            "lower": self.lower,
            "upper": self.upper,
            "confidence": self.confidence,
            "samples": self.samples,
        }


def bootstrap_confidence_interval(
    values: Iterable[float],
    statistic: Callable[[np.ndarray], float] | None = None,
    *,
    confidence: float = 0.95,
    iterations: int = 1000,
    random_seed: int = 42,
) -> ConfidenceInterval:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return ConfidenceInterval(0.0, 0.0, 0.0, confidence, 0)
    statistic = statistic or (lambda data: float(np.mean(data)))
    estimate = float(statistic(array))
    if array.size == 1 or iterations <= 1:
        return ConfidenceInterval(estimate, estimate, estimate, confidence, int(array.size))
    rng = np.random.default_rng(random_seed)
    draws = np.empty(iterations, dtype=float)
    for index in range(iterations):
        sample = rng.choice(array, size=array.size, replace=True)
        draws[index] = float(statistic(sample))
    alpha = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(draws, [alpha, 1.0 - alpha])
    return ConfidenceInterval(estimate, float(lower), float(upper), confidence, int(array.size))


def bootstrap_difference_interval(
    left: Iterable[float],
    right: Iterable[float],
    *,
    confidence: float = 0.95,
    iterations: int = 1000,
    random_seed: int = 42,
) -> ConfidenceInterval:
    left_array = np.asarray(list(left), dtype=float)
    right_array = np.asarray(list(right), dtype=float)
    if left_array.shape != right_array.shape:
        raise ValueError("paired bootstrap requires equally sized arrays")
    mask = np.isfinite(left_array) & np.isfinite(right_array)
    differences = left_array[mask] - right_array[mask]
    return bootstrap_confidence_interval(
        differences,
        confidence=confidence,
        iterations=iterations,
        random_seed=random_seed,
    )



def paired_randomization_pvalue(
    left: Iterable[float],
    right: Iterable[float],
    *,
    alternative: str = "two-sided",
    iterations: int = 20000,
    random_seed: int = 42,
) -> float:
    """Paired sign-flip randomization test for a mean difference."""
    left_array = np.asarray(list(left), dtype=float)
    right_array = np.asarray(list(right), dtype=float)
    if left_array.shape != right_array.shape:
        raise ValueError("paired randomization requires equally sized arrays")
    mask = np.isfinite(left_array) & np.isfinite(right_array)
    differences = left_array[mask] - right_array[mask]
    if differences.size == 0:
        return 1.0
    if alternative not in {"two-sided", "greater", "less"}:
        raise ValueError("alternative must be two-sided, greater or less")

    observed = float(np.mean(differences))

    def is_extreme(value: float) -> bool:
        if alternative == "greater":
            return value >= observed - 1e-15
        if alternative == "less":
            return value <= observed + 1e-15
        return abs(value) >= abs(observed) - 1e-15

    n = int(differences.size)
    extreme = 0
    total = 0
    if n <= 18:
        for mask_bits in range(1 << n):
            signs = np.ones(n, dtype=float)
            for index in range(n):
                if mask_bits & (1 << index):
                    signs[index] = -1.0
            extreme += int(is_extreme(float(np.mean(differences * signs))))
            total += 1
    else:
        rng = np.random.default_rng(random_seed)
        total = max(1000, int(iterations))
        for _ in range(total):
            signs = rng.choice(np.asarray([-1.0, 1.0]), size=n)
            extreme += int(is_extreme(float(np.mean(differences * signs))))
    return float((extreme + 1) / (total + 1))


def holm_adjusted_pvalues(p_values: Iterable[float]) -> list[float]:
    values = np.asarray(list(p_values), dtype=float)
    if values.size == 0:
        return []
    if np.any(~np.isfinite(values)) or np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("p-values must be finite values in [0, 1]")
    order = np.argsort(values, kind="stable")
    adjusted = np.empty(values.size, dtype=float)
    running = 0.0
    m = int(values.size)
    for rank, index in enumerate(order):
        candidate = float((m - rank) * values[index])
        running = max(running, candidate)
        adjusted[index] = min(running, 1.0)
    return [float(value) for value in adjusted]


def paired_standardized_effect(left: Iterable[float], right: Iterable[float]) -> float:
    """Cohen's dz for paired measurements (left - right)."""
    left_array = np.asarray(list(left), dtype=float)
    right_array = np.asarray(list(right), dtype=float)
    if left_array.shape != right_array.shape:
        raise ValueError("paired effect size requires equally sized arrays")
    mask = np.isfinite(left_array) & np.isfinite(right_array)
    differences = left_array[mask] - right_array[mask]
    if differences.size < 2:
        return 0.0
    std = float(np.std(differences, ddof=1))
    if std <= 1e-12:
        return 0.0 if abs(float(np.mean(differences))) <= 1e-12 else float(np.sign(np.mean(differences)) * np.inf)
    return float(np.mean(differences) / std)

def quantile_summary(values: Iterable[float]) -> dict[str, float | int]:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {
            "count": 0,
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "p25": 0.0,
            "median": 0.0,
            "p75": 0.0,
            "max": 0.0,
        }
    q25, median, q75 = np.quantile(array, [0.25, 0.5, 0.75])
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "min": float(np.min(array)),
        "p25": float(q25),
        "median": float(median),
        "p75": float(q75),
        "max": float(np.max(array)),
    }


def normalized_entropy(counts: Iterable[int | float]) -> float:
    values = np.asarray(list(counts), dtype=float)
    values = values[np.isfinite(values) & (values > 0)]
    if values.size <= 1:
        return 0.0
    probabilities = values / values.sum()
    entropy = -float(np.sum(probabilities * np.log(probabilities)))
    return float(entropy / log(values.size))


def gini_coefficient(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return 0.0
    minimum = float(np.min(array))
    if minimum < 0:
        array = array - minimum
    if np.allclose(array, 0.0):
        return 0.0
    array = np.sort(array)
    n = array.size
    index = np.arange(1, n + 1)
    return float((2.0 * np.sum(index * array) / (n * np.sum(array))) - (n + 1.0) / n)


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator) or denominator == 0:
        return float(default)
    return float(numerator / denominator)


def weighted_mean(values: Iterable[float], weights: Iterable[float]) -> float:
    value_array = np.asarray(list(values), dtype=float)
    weight_array = np.asarray(list(weights), dtype=float)
    if value_array.shape != weight_array.shape:
        raise ValueError("values and weights must have equal shape")
    mask = np.isfinite(value_array) & np.isfinite(weight_array) & (weight_array >= 0)
    value_array = value_array[mask]
    weight_array = weight_array[mask]
    if value_array.size == 0 or weight_array.sum() <= 0:
        return 0.0
    return float(np.average(value_array, weights=weight_array))


def rank_percentile(values: Iterable[float], value: float) -> float:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return 0.5
    return float(np.mean(array <= value))
