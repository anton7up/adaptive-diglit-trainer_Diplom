from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sqlalchemy.orm import Session

from .. import models
from ..config import get_settings
from .inference import predict_tasks


@dataclass(frozen=True)
class PlannedTask:
    task_id: int
    skill_id: int
    skill_code: str
    skill_name: str
    probability: float
    difficulty: float
    planning_score: float
    reasons: dict[str, float]


def build_practice_plan(
    db: Session,
    user_id: int,
    length: int = 8,
    skill_id: int | None = None,
) -> dict[str, Any]:
    length = max(3, min(length, 20))
    query = db.query(models.Task).join(models.Skill).filter(models.Task.active.is_(True), models.Skill.active.is_(True))
    if skill_id is not None:
        query = query.filter(models.Task.skill_id == skill_id)
    tasks = query.all()
    if not tasks:
        return {"tasks": [], "summary": {"length": 0}}

    predictions = predict_tasks(db, user_id, tasks)
    target = get_settings().target_success_probability
    attempts = db.query(models.Attempt).filter(models.Attempt.user_id == user_id).all()
    exposure: dict[int, int] = {}
    skill_outcomes: dict[int, list[int]] = {}
    for attempt in attempts:
        exposure[attempt.task_id] = exposure.get(attempt.task_id, 0) + 1
        skill_outcomes.setdefault(attempt.task.skill_id, []).append(int(attempt.is_correct))

    skill_mastery = {
        sid: (sum(values) + 1.5) / (len(values) + 3.0)
        for sid, values in skill_outcomes.items()
    }
    selected: list[PlannedTask] = []
    used_ids: set[int] = set()
    skill_counts: dict[int, int] = {}

    for position in range(length):
        best = None
        best_score = -10**9
        best_reasons = None
        for prediction in predictions:
            task = prediction.task
            if task.id in used_ids:
                continue
            p = prediction.probability
            zone = max(0.0, 1.0 - abs(p - target) / 0.45)
            weakness = 1.0 - skill_mastery.get(task.skill_id, 0.5)
            novelty = 1.0 / (1.0 + exposure.get(task.id, 0))
            skill_balance = 1.0 / (1.0 + skill_counts.get(task.skill_id, 0))
            sequence_fit = 1.0
            if selected:
                previous = selected[-1]
                if previous.skill_id == task.skill_id and skill_id is None:
                    sequence_fit -= 0.25
                if abs(previous.difficulty - task.difficulty) > 0.35:
                    sequence_fit -= 0.10
            score = 0.38 * zone + 0.24 * weakness + 0.16 * novelty + 0.14 * skill_balance + 0.08 * sequence_fit
            if score > best_score:
                best = prediction
                best_score = score
                best_reasons = {
                    "learning_zone": zone,
                    "skill_weakness": weakness,
                    "novelty": novelty,
                    "skill_balance": skill_balance,
                    "sequence_fit": sequence_fit,
                }
        if best is None:
            break
        task = best.task
        used_ids.add(task.id)
        skill_counts[task.skill_id] = skill_counts.get(task.skill_id, 0) + 1
        selected.append(PlannedTask(
            task_id=task.id,
            skill_id=task.skill_id,
            skill_code=task.skill.code,
            skill_name=task.skill.name,
            probability=best.probability,
            difficulty=task.difficulty,
            planning_score=float(best_score),
            reasons={key: float(value) for key, value in (best_reasons or {}).items()},
        ))

    probabilities = [item.probability for item in selected]
    return {
        "tasks": [item.__dict__ for item in selected],
        "summary": {
            "length": len(selected),
            "mean_predicted_success": sum(probabilities) / len(probabilities) if probabilities else 0.0,
            "target_success": target,
            "skills_covered": len({item.skill_id for item in selected}),
            "model_version": predictions[0].model_version if predictions else None,
        },
    }
