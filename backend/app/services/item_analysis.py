from __future__ import annotations

from collections import Counter, defaultdict
from math import sqrt
from typing import Any

from sqlalchemy.orm import Session

from .. import models


def _point_biserial(item_outcomes: list[int], total_scores: list[float]) -> float | None:
    n = len(item_outcomes)
    if n < 8 or len(set(item_outcomes)) < 2:
        return None
    positives = [score for outcome, score in zip(item_outcomes, total_scores) if outcome == 1]
    negatives = [score for outcome, score in zip(item_outcomes, total_scores) if outcome == 0]
    if not positives or not negatives:
        return None
    mean_pos = sum(positives) / len(positives)
    mean_neg = sum(negatives) / len(negatives)
    mean_all = sum(total_scores) / n
    variance = sum((value - mean_all) ** 2 for value in total_scores) / max(n - 1, 1)
    sd = sqrt(variance)
    if sd <= 1e-12:
        return None
    p = len(positives) / n
    q = 1.0 - p
    return (mean_pos - mean_neg) / sd * sqrt(p * q)


def analyze_item_bank(db: Session, include_synthetic: bool = False) -> list[dict[str, Any]]:
    query = db.query(models.Attempt).join(models.User)
    if not include_synthetic:
        query = query.filter(~models.User.username.like("synthetic_%"))
    attempts = query.all()
    by_user: dict[int, list[models.Attempt]] = defaultdict(list)
    by_task: dict[int, list[models.Attempt]] = defaultdict(list)
    for attempt in attempts:
        by_user[attempt.user_id].append(attempt)
        by_task[attempt.task_id].append(attempt)

    user_accuracy = {
        user_id: sum(int(item.is_correct) for item in rows) / len(rows)
        for user_id, rows in by_user.items() if rows
    }

    output = []
    for task in db.query(models.Task).order_by(models.Task.skill_id, models.Task.id).all():
        rows = by_task.get(task.id, [])
        n = len(rows)
        correct = sum(int(item.is_correct) for item in rows)
        empirical_success = correct / n if n else None
        distractors = Counter(item.selected_option for item in rows if not item.is_correct)
        response_times = sorted(item.response_time_ms for item in rows if item.response_time_ms > 0)
        median_time = None
        if response_times:
            mid = len(response_times) // 2
            median_time = (
                response_times[mid] if len(response_times) % 2
                else (response_times[mid - 1] + response_times[mid]) / 2
            )
        item_outcomes = [int(item.is_correct) for item in rows if item.user_id in user_accuracy]
        totals = [user_accuracy[item.user_id] for item in rows if item.user_id in user_accuracy]
        discrimination = _point_biserial(item_outcomes, totals)
        difficulty_gap = (empirical_success - (1.0 - task.difficulty)) if empirical_success is not None else None
        flags = []
        if n >= 10 and empirical_success is not None and empirical_success > 0.95:
            flags.append("too_easy")
        if n >= 10 and empirical_success is not None and empirical_success < 0.15:
            flags.append("too_hard")
        if discrimination is not None and discrimination < 0.05:
            flags.append("low_discrimination")
        if n >= 10 and not distractors:
            flags.append("no_distractor_usage")
        output.append({
            "task_id": task.id,
            "skill_id": task.skill_id,
            "skill_code": task.skill.code,
            "question": task.question,
            "configured_difficulty": task.difficulty,
            "attempts": n,
            "empirical_success_rate": empirical_success,
            "difficulty_alignment_gap": difficulty_gap,
            "point_biserial_discrimination": discrimination,
            "median_response_time_ms": median_time,
            "distractor_counts": dict(distractors),
            "flags": flags,
        })
    return output


def summarize_item_bank(rows: list[dict[str, Any]]) -> dict[str, Any]:
    observed = [row for row in rows if row["attempts"] > 0]
    flagged = [row for row in rows if row["flags"]]
    discriminations = [row["point_biserial_discrimination"] for row in observed if row["point_biserial_discrimination"] is not None]
    return {
        "items_total": len(rows),
        "items_with_observations": len(observed),
        "items_flagged": len(flagged),
        "mean_discrimination": sum(discriminations) / len(discriminations) if discriminations else None,
        "flags": {
            flag: sum(flag in row["flags"] for row in rows)
            for flag in {flag for row in rows for flag in row["flags"]}
        },
    }
