from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sqlalchemy.orm import Session

from .. import models
from ..timeutils import utcnow_naive
from .feature_engineering import build_candidate_frame
from .model_registry import ModelRegistry


@dataclass(frozen=True)
class TaskPrediction:
    task: models.Task
    probability: float
    model_version: str
    features: dict[str, object]


def predict_tasks(
    db: Session,
    user_id: int,
    tasks: list[models.Task],
    estimator=None,
    *,
    model_version: str | None = None,
) -> list[TaskPrediction]:
    if not tasks:
        return []

    resolved_estimator = estimator
    resolved_version = model_version
    if resolved_estimator is None:
        registry = ModelRegistry(db)
        record, resolved_estimator = registry.load_active()
        resolved_version = record.version if record is not None else None

    frame = build_candidate_frame(db, user_id, tasks, utcnow_naive())
    if resolved_estimator is None:
        probabilities = _fallback_probabilities(frame)
        resolved_version = "cold-start-baseline"
    else:
        probabilities = resolved_estimator.predict_proba(frame)[:, 1]
        resolved_version = resolved_version or "external-estimator"

    output: list[TaskPrediction] = []
    for index, task in enumerate(tasks):
        probability = float(np.clip(probabilities[index], 0.01, 0.99))
        output.append(TaskPrediction(task, probability, resolved_version, frame.iloc[index].to_dict()))
    return output


def _fallback_probabilities(frame):
    """Deterministic cold-start baseline used only before a trained artifact exists."""
    ability = 0.55 * frame["skill_accuracy"].to_numpy(dtype=float) + 0.45 * frame["user_accuracy_total"].to_numpy(dtype=float)
    difficulty = frame["task_difficulty"].to_numpy(dtype=float)
    logits = 3.2 * (ability - difficulty)
    return 1.0 / (1.0 + np.exp(-logits))
