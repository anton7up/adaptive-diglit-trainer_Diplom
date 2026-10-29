from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session, joinedload

from .. import models
from ..timeutils import utcnow_naive
from .calibration import expected_calibration_error
from .drift_monitor import compare_feature_drift
from .feature_engineering import build_training_frame
from .statistics_utils import quantile_summary


@dataclass(frozen=True)
class MonitoringAlert:
    code: str
    severity: str
    message: str
    value: Any
    threshold: Any

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
            "value": self.value,
            "threshold": self.threshold,
        }


def model_monitoring_report(
    db: Session,
    *,
    reference_days: int = 60,
    recent_days: int = 14,
    minimum_recent_rows: int = 40,
) -> dict[str, Any]:
    now = utcnow_naive()
    attempts = (
        db.query(models.Attempt)
        .options(joinedload(models.Attempt.task).joinedload(models.Task.skill))
        .order_by(models.Attempt.created_at.asc(), models.Attempt.id.asc())
        .all()
    )
    frame = build_training_frame(attempts)
    if frame.empty:
        return {
            "generated_at": now.isoformat() + "Z",
            "rows": 0,
            "alerts": [MonitoringAlert("no_data", "critical", "No attempts are available for monitoring.", 0, ">0").as_dict()],
        }
    frame["created_at"] = pd.to_datetime(frame["created_at"])
    recent_start = now - timedelta(days=recent_days)
    reference_start = now - timedelta(days=reference_days + recent_days)
    reference_end = recent_start
    recent = frame[frame["created_at"] >= recent_start].copy()
    reference = frame[(frame["created_at"] >= reference_start) & (frame["created_at"] < reference_end)].copy()
    if len(reference) < minimum_recent_rows:
        cut = max(1, int(len(frame) * 0.70))
        reference = frame.iloc[:cut].copy()
        recent = frame.iloc[cut:].copy()

    alerts: list[MonitoringAlert] = []
    if len(recent) < minimum_recent_rows:
        alerts.append(MonitoringAlert(
            "recent_sample_small",
            "warning",
            "Recent monitoring window contains few observations; estimates can be unstable.",
            int(len(recent)),
            f">={minimum_recent_rows}",
        ))

    drift = compare_feature_drift(reference, recent) if len(reference) and len(recent) else []
    severe = [signal for signal in drift if signal.severity == "drift"]
    watch = [signal for signal in drift if signal.severity == "watch"]
    if severe:
        alerts.append(MonitoringAlert(
            "feature_drift",
            "critical",
            "Strong population stability shift detected in one or more features.",
            len(severe),
            0,
        ))
    elif watch:
        alerts.append(MonitoringAlert(
            "feature_drift_watch",
            "warning",
            "Moderate population stability shift detected.",
            len(watch),
            0,
        ))

    performance = _performance_from_attempts(db, recent_start)
    if performance["rows"] >= minimum_recent_rows:
        if performance["brier"] > 0.25:
            alerts.append(MonitoringAlert(
                "brier_degradation",
                "critical",
                "Recent probabilistic error exceeds the operational threshold.",
                performance["brier"],
                "<=0.25",
            ))
        if performance["ece"] > 0.12:
            alerts.append(MonitoringAlert(
                "calibration_degradation",
                "warning",
                "Recent predictions are weakly calibrated.",
                performance["ece"],
                "<=0.12",
            ))

    prediction_values = [float(row.predicted_probability) for row in attempts if row.predicted_probability is not None]
    response_values = [float(row.response_time_ms) / 1000.0 for row in attempts if row.response_time_ms is not None]
    return {
        "generated_at": now.isoformat() + "Z",
        "rows": int(len(frame)),
        "reference_rows": int(len(reference)),
        "recent_rows": int(len(recent)),
        "reference_days": reference_days,
        "recent_days": recent_days,
        "prediction_distribution": quantile_summary(prediction_values),
        "response_time_seconds": quantile_summary(response_values),
        "recent_performance": performance,
        "drift": [signal.__dict__ for signal in drift[:20]],
        "alerts": [alert.as_dict() for alert in alerts],
        "status": _overall_status(alerts),
    }


def _performance_from_attempts(db: Session, since) -> dict[str, Any]:
    rows = (
        db.query(models.Attempt)
        .filter(models.Attempt.created_at >= since, models.Attempt.predicted_probability.isnot(None))
        .all()
    )
    if not rows:
        return {"rows": 0, "brier": 0.0, "ece": 0.0, "mean_probability": 0.0, "observed_rate": 0.0}
    y = np.asarray([int(row.is_correct) for row in rows], dtype=int)
    p = np.asarray([float(row.predicted_probability) for row in rows], dtype=float)
    return {
        "rows": int(len(rows)),
        "brier": float(np.mean((p - y) ** 2)),
        "ece": expected_calibration_error(y, p, bins=8),
        "mean_probability": float(np.mean(p)),
        "observed_rate": float(np.mean(y)),
    }


def _overall_status(alerts: list[MonitoringAlert]) -> str:
    if any(item.severity == "critical" for item in alerts):
        return "critical"
    if alerts:
        return "warning"
    return "healthy"
