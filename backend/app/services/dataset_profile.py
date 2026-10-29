from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .feature_engineering import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET_COLUMN
from .statistics_utils import gini_coefficient, normalized_entropy, quantile_summary


def _finite_numeric(series: pd.Series) -> np.ndarray:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    return values[np.isfinite(values)]


def _target_correlation(series: pd.Series, target: pd.Series) -> float | None:
    x = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(target, errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 8:
        return None
    x = x[mask]
    y = y[mask]
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def numeric_feature_profile(frame: pd.DataFrame) -> list[dict[str, Any]]:
    target = frame[TARGET_COLUMN] if TARGET_COLUMN in frame else pd.Series(dtype=float)
    rows: list[dict[str, Any]] = []
    for name in NUMERIC_FEATURES:
        if name not in frame:
            rows.append({"feature": name, "present": False})
            continue
        raw = frame[name]
        values = _finite_numeric(raw)
        missing = int(raw.isna().sum()) + int((~np.isfinite(pd.to_numeric(raw, errors="coerce"))).sum() - raw.isna().sum())
        summary = quantile_summary(values) if len(values) else {}
        rows.append({
            "feature": name,
            "present": True,
            "rows": int(len(raw)),
            "missing": int(max(missing, 0)),
            "missing_rate": float(max(missing, 0) / max(len(raw), 1)),
            "unique": int(pd.Series(values).nunique()) if len(values) else 0,
            "mean": float(np.mean(values)) if len(values) else None,
            "std": float(np.std(values)) if len(values) else None,
            "minimum": float(np.min(values)) if len(values) else None,
            "maximum": float(np.max(values)) if len(values) else None,
            "quantiles": summary,
            "target_correlation": _target_correlation(raw, target) if len(target) else None,
        })
    return rows


def categorical_feature_profile(frame: pd.DataFrame, top_n: int = 12) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for name in CATEGORICAL_FEATURES:
        if name not in frame:
            output.append({"feature": name, "present": False})
            continue
        values = frame[name].fillna("<missing>").astype(str)
        counts = values.value_counts(dropna=False)
        output.append({
            "feature": name,
            "present": True,
            "unique": int(values.nunique()),
            "missing": int((values == "<missing>").sum()),
            "entropy": normalized_entropy(counts.to_numpy(dtype=float)),
            "top_values": [
                {
                    "value": str(key),
                    "count": int(count),
                    "share": float(count / max(len(values), 1)),
                }
                for key, count in counts.head(top_n).items()
            ],
        })
    return output


def exposure_profile(frame: pd.DataFrame, column: str) -> dict[str, Any]:
    if column not in frame or frame.empty:
        return {"entities": 0, "minimum": 0, "median": 0.0, "maximum": 0, "gini": 0.0, "top_share": 0.0}
    counts = frame[column].value_counts().to_numpy(dtype=float)
    total = float(counts.sum())
    return {
        "entities": int(len(counts)),
        "minimum": int(np.min(counts)) if len(counts) else 0,
        "median": float(np.median(counts)) if len(counts) else 0.0,
        "maximum": int(np.max(counts)) if len(counts) else 0,
        "gini": gini_coefficient(counts),
        "top_share": float(np.max(counts) / total) if total else 0.0,
    }


def temporal_profile(frame: pd.DataFrame) -> dict[str, Any]:
    if "created_at" not in frame or frame.empty:
        return {"start": None, "end": None, "days": 0.0, "rows_per_day": 0.0}
    values = pd.to_datetime(frame["created_at"], errors="coerce").dropna()
    if values.empty:
        return {"start": None, "end": None, "days": 0.0, "rows_per_day": 0.0}
    start = values.min()
    end = values.max()
    span_days = max(float((end - start).total_seconds() / 86400.0), 0.0)
    return {
        "start": start.isoformat(),
        "end": end.isoformat(),
        "days": span_days,
        "rows_per_day": float(len(values) / max(span_days, 1.0)),
    }


def suspicious_feature_correlations(frame: pd.DataFrame, threshold: float = 0.98) -> list[dict[str, Any]]:
    numeric = [name for name in NUMERIC_FEATURES if name in frame]
    if len(numeric) < 2:
        return []
    matrix = frame[numeric].apply(pd.to_numeric, errors="coerce").corr().abs()
    output: list[dict[str, Any]] = []
    for left_index, left in enumerate(numeric):
        for right in numeric[left_index + 1 :]:
            value = matrix.loc[left, right]
            if pd.notna(value) and float(value) >= threshold:
                output.append({"left": left, "right": right, "absolute_correlation": float(value)})
    return sorted(output, key=lambda row: row["absolute_correlation"], reverse=True)


def build_dataset_profile(frame: pd.DataFrame) -> dict[str, Any]:
    target = frame[TARGET_COLUMN].astype(int) if TARGET_COLUMN in frame and len(frame) else pd.Series(dtype=int)
    positive = float(target.mean()) if len(target) else 0.0
    users = exposure_profile(frame, "user_id")
    tasks = exposure_profile(frame, "task_id")
    numeric = numeric_feature_profile(frame)
    strongest = sorted(
        [row for row in numeric if row.get("target_correlation") is not None],
        key=lambda row: abs(float(row["target_correlation"])),
        reverse=True,
    )[:12]
    return {
        "rows": int(len(frame)),
        "features": int(len(NUMERIC_FEATURES) + len(CATEGORICAL_FEATURES)),
        "numeric_features": int(len(NUMERIC_FEATURES)),
        "categorical_features": int(len(CATEGORICAL_FEATURES)),
        "target": {
            "positive": int(target.sum()) if len(target) else 0,
            "negative": int(len(target) - target.sum()) if len(target) else 0,
            "positive_rate": positive,
            "minority_rate": min(positive, 1.0 - positive) if len(target) else 0.0,
        },
        "user_exposure": users,
        "task_exposure": tasks,
        "temporal": temporal_profile(frame),
        "numeric": numeric,
        "categorical": categorical_feature_profile(frame),
        "strongest_univariate_correlations": [
            {"feature": row["feature"], "target_correlation": row["target_correlation"]}
            for row in strongest
        ],
        "high_feature_correlations": suspicious_feature_correlations(frame),
        "interpretation": {
            "target_correlation": "Univariate diagnostic only; it is not a causal effect or feature-importance estimate.",
            "exposure_gini": "0 denotes balanced exposure; values closer to 1 indicate concentration on fewer entities.",
        },
    }
