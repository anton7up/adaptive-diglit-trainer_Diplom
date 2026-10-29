from __future__ import annotations

import math
from dataclasses import dataclass

from sqlalchemy.orm import Session

from .. import models
from ..config import get_settings
from .inference import TaskPrediction, predict_tasks


@dataclass(frozen=True)
class SelectionResult:
    prediction: TaskPrediction
    score: float
    components: dict[str, float]


def _recent_task_ids(db: Session, user_id: int, limit: int = 8) -> set[int]:
    return {
        int(row[0])
        for row in db.query(models.Attempt.task_id)
        .filter(models.Attempt.user_id == user_id)
        .order_by(models.Attempt.created_at.desc(), models.Attempt.id.desc())
        .limit(limit)
        .all()
    }


def _task_exposures(db: Session, user_id: int) -> dict[int, int]:
    rows = (
        db.query(models.Attempt.task_id)
        .filter(models.Attempt.user_id == user_id)
        .all()
    )
    counts: dict[int, int] = {}
    for (task_id,) in rows:
        counts[int(task_id)] = counts.get(int(task_id), 0) + 1
    return counts


def _selection_components(
    probability: float,
    *,
    skill_accuracy: float,
    exposure_count: int,
    recently_seen: bool,
    cognitive_level: int,
    target_probability: float,
) -> dict[str, float]:
    zone = math.exp(-((probability - target_probability) ** 2) / (2 * 0.16**2))
    weakness = 1.0 - skill_accuracy
    novelty = 1.0 / (1.0 + exposure_count)
    recency = -0.65 if recently_seen else 0.0
    uncertainty = 1.0 - abs(probability - 0.5) * 2.0
    expected_cognitive_level = 1 + round(2 * skill_accuracy)
    cognitive_fit = 1.0 - min(abs(cognitive_level - expected_cognitive_level) / 3.0, 1.0)
    return {
        "learning_zone": float(zone),
        "skill_weakness": float(weakness),
        "novelty": float(novelty),
        "uncertainty": float(uncertainty),
        "cognitive_fit": float(cognitive_fit),
        "recency_penalty": float(recency),
    }


def _selection_score(components: dict[str, float]) -> float:
    return float(
        0.46 * components["learning_zone"]
        + 0.20 * components["skill_weakness"]
        + 0.14 * components["novelty"]
        + 0.08 * components["uncertainty"]
        + 0.12 * components["cognitive_fit"]
        + components["recency_penalty"]
    )


def rank_tasks(
    db: Session,
    user_id: int,
    tasks: list[models.Task],
    estimator=None,
    *,
    limit: int | None = None,
) -> list[SelectionResult]:
    """Rank candidate tasks without mutating database state.

    The learned model supplies P(correct). The policy layer then balances the
    productive challenge zone, weak-skill coverage, novelty, prediction
    uncertainty, cognitive-level fit and a recent-exposure penalty. Keeping
    model inference separate from policy scoring makes both parts testable.
    """
    if not tasks:
        return []
    recent_ids = _recent_task_ids(db, user_id)
    exposures = _task_exposures(db, user_id)
    per_skill_stats = _skill_stats(db, user_id)
    target = get_settings().target_success_probability

    ranked: list[SelectionResult] = []
    for prediction in predict_tasks(db, user_id, tasks, estimator=estimator):
        task = prediction.task
        components = _selection_components(
            prediction.probability,
            skill_accuracy=per_skill_stats.get(task.skill_id, 0.5),
            exposure_count=exposures.get(task.id, 0),
            recently_seen=task.id in recent_ids,
            cognitive_level=task.cognitive_level,
            target_probability=target,
        )
        ranked.append(SelectionResult(prediction, _selection_score(components), components))

    ranked.sort(key=lambda item: (item.score, -item.prediction.task.id), reverse=True)
    return ranked if limit is None else ranked[: max(limit, 0)]


def choose_next_task(db: Session, user_id: int, skill_id: int | None = None) -> SelectionResult | None:
    query = db.query(models.Task).join(models.Skill).filter(models.Task.active.is_(True), models.Skill.active.is_(True))
    if skill_id is not None:
        query = query.filter(models.Task.skill_id == skill_id)
    tasks = query.all()
    ranked = rank_tasks(db, user_id, tasks, limit=1)
    if not ranked:
        return None

    winner = ranked[0]
    db.add(models.PredictionAudit(
        user_id=user_id,
        task_id=winner.prediction.task.id,
        model_version=winner.prediction.model_version,
        probability=winner.prediction.probability,
        feature_vector=winner.prediction.features,
        selection_score=winner.score,
        selection_components=winner.components,
    ))
    db.commit()
    return winner


def _skill_stats(db: Session, user_id: int) -> dict[int, float]:
    attempts = (
        db.query(models.Attempt)
        .join(models.Task)
        .filter(models.Attempt.user_id == user_id)
        .order_by(models.Attempt.created_at.asc(), models.Attempt.id.asc())
        .all()
    )
    totals: dict[int, list[int]] = {}
    for attempt in attempts:
        row = totals.setdefault(attempt.task.skill_id, [0, 0])
        row[0] += int(attempt.is_correct)
        row[1] += 1
    return {skill_id: (correct + 1.5) / (count + 3.0) for skill_id, (correct, count) in totals.items()}
