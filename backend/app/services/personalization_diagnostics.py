from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
from sqlalchemy import func
from sqlalchemy.orm import Session, joinedload

from .. import models
from .adaptive_engine import rank_tasks
from .feature_engineering import build_candidate_frame
from .statistics_utils import normalized_entropy, quantile_summary


def _active_tasks(db: Session) -> list[models.Task]:
    return (
        db.query(models.Task)
        .options(joinedload(models.Task.skill))
        .filter(models.Task.active.is_(True))
        .order_by(models.Task.id.asc())
        .all()
    )


def _eligible_students(db: Session, minimum_attempts: int) -> list[models.User]:
    attempt_counts = dict(
        db.query(models.Attempt.user_id, func.count(models.Attempt.id))
        .group_by(models.Attempt.user_id)
        .all()
    )
    return [
        user
        for user in db.query(models.User).filter(models.User.role == "student").order_by(models.User.id.asc()).all()
        if attempt_counts.get(user.id, 0) >= minimum_attempts
    ]


def probability_surface(db: Session, estimator, users: list[models.User], tasks: list[models.Task]) -> dict[str, Any]:
    all_probabilities: list[float] = []
    user_means: list[float] = []
    user_spreads: list[float] = []
    task_means: dict[int, list[float]] = {task.id: [] for task in tasks}
    for user in users:
        frame = build_candidate_frame(db, user.id, tasks)
        if frame.empty:
            continue
        probabilities = estimator.predict_proba(frame)[:, 1].astype(float)
        all_probabilities.extend(probabilities.tolist())
        user_means.append(float(np.mean(probabilities)))
        user_spreads.append(float(np.max(probabilities) - np.min(probabilities)))
        for task, value in zip(tasks, probabilities):
            task_means.setdefault(int(task.id), []).append(float(value))
    task_average_values = [float(np.mean(values)) for values in task_means.values() if values]
    return {
        "predictions": int(len(all_probabilities)),
        "probability": quantile_summary(np.asarray(all_probabilities)) if all_probabilities else {},
        "user_mean_probability": quantile_summary(np.asarray(user_means)) if user_means else {},
        "within_user_probability_spread": quantile_summary(np.asarray(user_spreads)) if user_spreads else {},
        "task_mean_probability": quantile_summary(np.asarray(task_average_values)) if task_average_values else {},
    }


def ranking_diversity(db: Session, estimator, users: list[models.User], tasks: list[models.Task], top_k: int = 5) -> dict[str, Any]:
    first_choice: Counter[int] = Counter()
    top_choices: Counter[int] = Counter()
    skill_choices: Counter[str] = Counter()
    pairwise_disagreements: list[float] = []
    rankings: list[list[int]] = []

    for user in users:
        ranked = rank_tasks(db, user.id, tasks, estimator, limit=max(top_k, 10))
        ids = [int(row.prediction.task.id) for row in ranked]
        if not ids:
            continue
        rankings.append(ids)
        first_choice[ids[0]] += 1
        for row in ranked[:top_k]:
            task = row.prediction.task
            top_choices[int(task.id)] += 1
            skill_choices[str(task.skill.code)] += 1

    for left_index, left in enumerate(rankings):
        left_set = set(left[:top_k])
        for right in rankings[left_index + 1 :]:
            right_set = set(right[:top_k])
            union = left_set | right_set
            similarity = len(left_set & right_set) / max(len(union), 1)
            pairwise_disagreements.append(1.0 - similarity)

    total_top = sum(top_choices.values())
    total_first = sum(first_choice.values())
    return {
        "users_ranked": int(len(rankings)),
        "unique_first_choices": int(len(first_choice)),
        "unique_top_k_tasks": int(len(top_choices)),
        "first_choice_entropy": normalized_entropy(np.asarray(list(first_choice.values()), dtype=float)) if total_first else 0.0,
        "top_k_task_entropy": normalized_entropy(np.asarray(list(top_choices.values()), dtype=float)) if total_top else 0.0,
        "pairwise_top_k_disagreement": float(np.mean(pairwise_disagreements)) if pairwise_disagreements else 0.0,
        "skill_distribution": [
            {"skill": skill, "count": int(count), "share": float(count / max(total_top, 1))}
            for skill, count in sorted(skill_choices.items())
        ],
        "most_common_first_choices": [
            {"task_id": int(task_id), "count": int(count), "share": float(count / max(total_first, 1))}
            for task_id, count in first_choice.most_common(10)
        ],
    }


def challenge_zone_diagnostics(db: Session, estimator, users: list[models.User], tasks: list[models.Task]) -> dict[str, Any]:
    selected_probabilities: list[float] = []
    selected_scores: list[float] = []
    in_zone = 0
    for user in users:
        ranked = rank_tasks(db, user.id, tasks, estimator, limit=1)
        if not ranked:
            continue
        top = ranked[0]
        probability = float(top.prediction.probability)
        selected_probabilities.append(probability)
        selected_scores.append(float(top.score))
        if 0.45 <= probability <= 0.80:
            in_zone += 1
    return {
        "selections": int(len(selected_probabilities)),
        "productive_zone": [0.45, 0.80],
        "productive_zone_share": float(in_zone / max(len(selected_probabilities), 1)),
        "selected_probability": quantile_summary(np.asarray(selected_probabilities)) if selected_probabilities else {},
        "adaptive_score": quantile_summary(np.asarray(selected_scores)) if selected_scores else {},
    }


def personalization_report(
    db: Session,
    estimator,
    *,
    max_users: int = 24,
    minimum_attempts: int = 8,
    top_k: int = 5,
) -> dict[str, Any]:
    tasks = _active_tasks(db)
    users = _eligible_students(db, minimum_attempts)[:max_users]
    if not tasks:
        raise ValueError("no active tasks")
    if len(users) < 2:
        raise ValueError("at least two students with sufficient history are required")
    return {
        "users": int(len(users)),
        "tasks": int(len(tasks)),
        "minimum_attempts": int(minimum_attempts),
        "probability_surface": probability_surface(db, estimator, users, tasks),
        "ranking_diversity": ranking_diversity(db, estimator, users, tasks, top_k=top_k),
        "challenge_zone": challenge_zone_diagnostics(db, estimator, users, tasks),
        "interpretation": (
            "A non-zero probability spread and ranking disagreement demonstrate that the model produces "
            "user-dependent task rankings rather than one global task order."
        ),
    }
