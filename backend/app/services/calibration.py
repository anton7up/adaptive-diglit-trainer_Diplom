from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss


@dataclass(frozen=True)
class CalibrationBin:
    lower: float
    upper: float
    count: int
    mean_probability: float
    observed_rate: float
    absolute_gap: float

    def as_dict(self) -> dict[str, float | int]:
        return {
            "lower": self.lower,
            "upper": self.upper,
            "count": self.count,
            "mean_probability": self.mean_probability,
            "observed_rate": self.observed_rate,
            "absolute_gap": self.absolute_gap,
        }


def _validate(y_true, probability) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probability, dtype=float)
    if y.shape != p.shape:
        raise ValueError("y_true and probability must have equal shape")
    if y.ndim != 1:
        raise ValueError("calibration input must be one-dimensional")
    if y.size == 0:
        raise ValueError("calibration input is empty")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("y_true must contain binary labels")
    if not np.isfinite(p).all():
        raise ValueError("probability contains non-finite values")
    return y, np.clip(p, 1e-6, 1.0 - 1e-6)


def reliability_bins(y_true, probability, bins: int = 10) -> list[CalibrationBin]:
    y, p = _validate(y_true, probability)
    bins = max(2, min(int(bins), 50))
    edges = np.linspace(0.0, 1.0, bins + 1)
    output: list[CalibrationBin] = []
    for index in range(bins):
        lower = float(edges[index])
        upper = float(edges[index + 1])
        if index == bins - 1:
            mask = (p >= lower) & (p <= upper)
        else:
            mask = (p >= lower) & (p < upper)
        count = int(mask.sum())
        if count == 0:
            continue
        mean_probability = float(np.mean(p[mask]))
        observed_rate = float(np.mean(y[mask]))
        output.append(CalibrationBin(
            lower=lower,
            upper=upper,
            count=count,
            mean_probability=mean_probability,
            observed_rate=observed_rate,
            absolute_gap=abs(mean_probability - observed_rate),
        ))
    return output


def expected_calibration_error(y_true, probability, bins: int = 10) -> float:
    y, _ = _validate(y_true, probability)
    rows = reliability_bins(y_true, probability, bins=bins)
    return float(sum((row.count / len(y)) * row.absolute_gap for row in rows))


def maximum_calibration_error(y_true, probability, bins: int = 10) -> float:
    rows = reliability_bins(y_true, probability, bins=bins)
    return float(max((row.absolute_gap for row in rows), default=0.0))


def calibration_slope_intercept(y_true, probability) -> dict[str, float]:
    y, p = _validate(y_true, probability)
    if len(np.unique(y)) < 2:
        return {"intercept": 0.0, "slope": 0.0}
    logits = np.log(p / (1.0 - p)).reshape(-1, 1)
    model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000)
    model.fit(logits, y)
    return {
        "intercept": float(model.intercept_[0]),
        "slope": float(model.coef_[0][0]),
    }


def brier_decomposition(y_true, probability, bins: int = 10) -> dict[str, float]:
    y, p = _validate(y_true, probability)
    rows = reliability_bins(y, p, bins=bins)
    prevalence = float(np.mean(y))
    uncertainty = prevalence * (1.0 - prevalence)
    reliability = 0.0
    resolution = 0.0
    n = len(y)
    for row in rows:
        weight = row.count / n
        reliability += weight * (row.mean_probability - row.observed_rate) ** 2
        resolution += weight * (row.observed_rate - prevalence) ** 2
    brier = float(brier_score_loss(y, p))
    return {
        "brier": brier,
        "reliability": float(reliability),
        "resolution": float(resolution),
        "uncertainty": float(uncertainty),
        "identity_residual": float(brier - (reliability - resolution + uncertainty)),
    }


def calibration_report(y_true, probability, bins: int = 10) -> dict:
    y, p = _validate(y_true, probability)
    slope = calibration_slope_intercept(y, p)
    rows = reliability_bins(y, p, bins=bins)
    return {
        "samples": int(len(y)),
        "positive_rate": float(np.mean(y)),
        "mean_probability": float(np.mean(p)),
        "ece": expected_calibration_error(y, p, bins=bins),
        "mce": maximum_calibration_error(y, p, bins=bins),
        "log_loss": float(log_loss(y, np.column_stack([1.0 - p, p]), labels=[0, 1])),
        "slope": slope["slope"],
        "intercept": slope["intercept"],
        "brier_decomposition": brier_decomposition(y, p, bins=bins),
        "bins": [row.as_dict() for row in rows],
    }
