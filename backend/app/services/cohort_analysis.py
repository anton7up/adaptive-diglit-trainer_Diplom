from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from .. import models
from .calibration import expected_calibration_error
from .feature_engineering import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET_COLUMN
from .model_training import probability_metrics
from .statistics_utils import bootstrap_confidence_interval


@dataclass(frozen=True)
class CohortMetrics:
    cohort: str
    rows: int
    users: int
    positive_rate: float
    mean_probability: float
    brier: float
    roc_auc: float
    pr_auc: float
    ece: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "cohort": self.cohort,
            "rows": self.rows,
            "users": self.users,
            "positive_rate": self.positive_rate,
            "mean_probability": self.mean_probability,
            "brier": self.brier,
            "roc_auc": self.roc_auc,
            "pr_auc": self.pr_auc,
            "ece": self.ece,
        }


def attach_group_labels(db: Session, frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.assign(group_name=pd.Series(dtype=str))
    ids = [int(value) for value in frame["user_id"].dropna().unique()]
    users = db.query(models.User).filter(models.User.id.in_(ids)).all() if ids else []
    mapping = {user.id: (user.group_name or "UNSPECIFIED") for user in users}
    enriched = frame.copy()
    enriched["group_name"] = enriched["user_id"].map(mapping).fillna("UNSPECIFIED")
    return enriched


def evaluate_cohorts(
    estimator,
    frame: pd.DataFrame,
    *,
    cohort_column: str = "group_name",
    min_rows: int = 30,
    min_users: int = 2,
) -> dict[str, Any]:
    if cohort_column not in frame.columns:
        raise ValueError(f"missing cohort column: {cohort_column}")
    features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    rows: list[CohortMetrics] = []
    excluded: list[dict[str, Any]] = []
    for cohort, subset in frame.groupby(cohort_column, dropna=False):
        user_count = int(subset["user_id"].nunique())
        if len(subset) < min_rows or user_count < min_users:
            excluded.append({
                "cohort": str(cohort),
                "rows": int(len(subset)),
                "users": user_count,
                "reason": "insufficient support",
            })
            continue
        y = subset[TARGET_COLUMN].astype(int).to_numpy()
        probability = estimator.predict_proba(subset[features])[:, 1]
        metrics = probability_metrics(y, probability)
        rows.append(CohortMetrics(
            cohort=str(cohort),
            rows=int(len(subset)),
            users=user_count,
            positive_rate=float(np.mean(y)),
            mean_probability=float(np.mean(probability)),
            brier=metrics["brier"],
            roc_auc=metrics["roc_auc"],
            pr_auc=metrics["pr_auc"],
            ece=expected_calibration_error(y, probability),
        ))
    return {
        "cohort_column": cohort_column,
        "minimum_rows": min_rows,
        "minimum_users": min_users,
        "cohorts": [row.as_dict() for row in rows],
        "excluded": excluded,
        "parity": _parity_summary(rows),
        "interpretation": (
            "Cohort metrics are diagnostic only. Differences can reflect preparation level, task exposure or sampling; "
            "they are not evidence of causal bias without a controlled study."
        ),
    }


def _parity_summary(rows: list[CohortMetrics]) -> dict[str, Any]:
    if len(rows) < 2:
        return {
            "available": False,
            "reason": "at least two sufficiently supported cohorts are required",
        }
    metrics = {
        "brier": [row.brier for row in rows],
        "ece": [row.ece for row in rows],
        "roc_auc": [row.roc_auc for row in rows],
        "positive_rate": [row.positive_rate for row in rows],
        "mean_probability": [row.mean_probability for row in rows],
    }
    gaps = {name: float(max(values) - min(values)) for name, values in metrics.items()}
    calibration_bias = [row.mean_probability - row.positive_rate for row in rows]
    interval = bootstrap_confidence_interval(calibration_bias, iterations=800)
    return {
        "available": True,
        "max_min_gaps": gaps,
        "mean_calibration_bias_across_cohorts": interval.as_dict(),
        "largest_brier_cohort": max(rows, key=lambda row: row.brier).cohort,
        "smallest_brier_cohort": min(rows, key=lambda row: row.brier).cohort,
    }
