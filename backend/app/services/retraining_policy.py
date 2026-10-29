from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timedelta

from sqlalchemy.orm import Session

from .. import models
from ..timeutils import utcnow_naive


@dataclass(frozen=True)
class RetrainingDecision:
    should_retrain: bool
    reasons: list[str]
    new_attempts_since_training: int
    hours_since_training: float | None
    recent_brier: float | None
    recent_calibration_gap: float | None

    def as_dict(self):
        return asdict(self)


def evaluate_retraining_need(
    db: Session,
    min_new_attempts: int = 300,
    max_age_hours: int = 24 * 14,
    brier_threshold: float = 0.25,
    calibration_gap_threshold: float = 0.12,
) -> RetrainingDecision:
    active = (
        db.query(models.MLModelVersion)
        .filter(models.MLModelVersion.is_active.is_(True))
        .order_by(models.MLModelVersion.trained_at.desc())
        .first()
    )
    reasons: list[str] = []
    if active is None:
        return RetrainingDecision(True, ["no_active_model"], db.query(models.Attempt).count(), None, None, None)

    new_attempts = db.query(models.Attempt).filter(models.Attempt.created_at > active.trained_at).count()
    age_hours = max(0.0, (utcnow_naive() - active.trained_at).total_seconds() / 3600.0)
    if new_attempts >= min_new_attempts:
        reasons.append("enough_new_attempts")
    if age_hours >= max_age_hours:
        reasons.append("model_age_limit")

    recent = (
        db.query(models.Attempt)
        .filter(
            models.Attempt.prediction_model == active.version,
            models.Attempt.predicted_probability.isnot(None),
            models.Attempt.created_at >= utcnow_naive() - timedelta(days=30),
        )
        .all()
    )
    recent_brier = None
    recent_gap = None
    if len(recent) >= 30:
        recent_brier = sum((float(item.predicted_probability) - int(item.is_correct)) ** 2 for item in recent) / len(recent)
        mean_prediction = sum(float(item.predicted_probability) for item in recent) / len(recent)
        observed = sum(int(item.is_correct) for item in recent) / len(recent)
        recent_gap = abs(mean_prediction - observed)
        if recent_brier >= brier_threshold:
            reasons.append("brier_degradation")
        if recent_gap >= calibration_gap_threshold:
            reasons.append("calibration_degradation")

    return RetrainingDecision(bool(reasons), reasons, new_attempts, age_hours, recent_brier, recent_gap)
