from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from .feature_engineering import NUMERIC_FEATURES


@dataclass(frozen=True)
class DriftSignal:
    feature: str
    psi: float
    severity: str


def _safe_distribution(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    counts, _ = np.histogram(values, bins=edges)
    distribution = counts.astype(float) / max(counts.sum(), 1)
    return np.clip(distribution, 1e-6, 1.0)


def population_stability_index(reference: Iterable[float], current: Iterable[float], buckets: int = 10) -> float:
    ref = np.asarray(list(reference), dtype=float)
    cur = np.asarray(list(current), dtype=float)
    ref = ref[np.isfinite(ref)]
    cur = cur[np.isfinite(cur)]
    if len(ref) < 2 or len(cur) < 2:
        return 0.0
    quantiles = np.linspace(0.0, 1.0, buckets + 1)
    edges = np.unique(np.quantile(ref, quantiles))
    if len(edges) < 3:
        minimum = min(ref.min(), cur.min())
        maximum = max(ref.max(), cur.max())
        if math.isclose(minimum, maximum):
            return 0.0
        edges = np.linspace(minimum, maximum, buckets + 1)
    edges[0] = min(edges[0], cur.min()) - 1e-9
    edges[-1] = max(edges[-1], cur.max()) + 1e-9
    ref_dist = _safe_distribution(ref, edges)
    cur_dist = _safe_distribution(cur, edges)
    return float(np.sum((cur_dist - ref_dist) * np.log(cur_dist / ref_dist)))


def drift_severity(psi: float) -> str:
    if psi < 0.10:
        return "stable"
    if psi < 0.25:
        return "watch"
    return "drift"


def compare_feature_drift(reference: pd.DataFrame, current: pd.DataFrame) -> list[DriftSignal]:
    signals = []
    for feature in NUMERIC_FEATURES:
        if feature not in reference.columns or feature not in current.columns:
            continue
        psi = population_stability_index(reference[feature], current[feature])
        signals.append(DriftSignal(feature=feature, psi=psi, severity=drift_severity(psi)))
    return sorted(signals, key=lambda item: item.psi, reverse=True)
