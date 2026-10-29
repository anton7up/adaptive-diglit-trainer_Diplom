from __future__ import annotations

from datetime import datetime

from sqlalchemy.orm import Session, joinedload

from .. import models
from ..timeutils import utcnow_naive
from .feature_engineering import build_training_frame
from .data_quality import assert_training_quality
from .model_registry import ModelRegistry
from .model_training import train_and_compare


def train_from_database(db: Session, activate: bool = True) -> dict:
    run = models.MLTrainingRun(run_id="pending", status="running", started_at=utcnow_naive())
    db.add(run)
    db.flush()
    attempts = (
        db.query(models.Attempt)
        .options(joinedload(models.Attempt.task).joinedload(models.Task.skill))
        .order_by(models.Attempt.created_at.asc(), models.Attempt.id.asc())
        .all()
    )
    frame = build_training_frame(attempts)
    try:
        quality = assert_training_quality(frame)
        result = train_and_compare(frame)
        run.run_id = result["run_id"]
        run.finished_at = utcnow_naive()
        run.status = "completed"
        run.dataset_rows = len(frame)
        run.dataset_positive_rate = float(frame["target"].mean())
        run.feature_count = len(result["metadata"]["feature_names"])
        run.best_model_name = result["model_name"]
        run.best_model_version = result["version"]
        run.metrics = {item["name"]: item for item in result["metadata"]["candidates"]}
        run.parameters = {"candidates": [item["name"] for item in result["metadata"]["candidates"]], "quality": quality.as_dict()}
        ModelRegistry(db).register(result, activate=activate)
        db.commit()
        return result
    except Exception as exc:
        run.finished_at = utcnow_naive()
        run.status = "failed"
        run.notes = str(exc)
        db.commit()
        raise
