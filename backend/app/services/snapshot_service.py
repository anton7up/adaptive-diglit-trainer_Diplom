from __future__ import annotations

from datetime import datetime

from sqlalchemy.orm import Session

from .. import models
from ..timeutils import utcnow_naive


def _accuracy(items: list[models.Attempt]) -> float:
    return sum(int(item.is_correct) for item in items) / len(items) if items else 0.5


def refresh_user_skill_snapshot(db: Session, user_id: int, skill_id: int) -> models.UserSkillSnapshot:
    attempts = (
        db.query(models.Attempt)
        .join(models.Task)
        .filter(models.Attempt.user_id == user_id, models.Task.skill_id == skill_id)
        .order_by(models.Attempt.created_at.asc(), models.Attempt.id.asc())
        .all()
    )
    snapshot = db.query(models.UserSkillSnapshot).filter_by(user_id=user_id, skill_id=skill_id).first()
    if snapshot is None:
        snapshot = models.UserSkillSnapshot(user_id=user_id, skill_id=skill_id)
        db.add(snapshot)
    outcomes = [int(item.is_correct) for item in attempts]
    times = [item.response_time_ms for item in attempts if item.response_time_ms > 0]
    last = attempts[-1] if attempts else None
    streak_correct = 0
    streak_incorrect = 0
    for outcome in reversed(outcomes):
        if outcome and streak_incorrect == 0:
            streak_correct += 1
        elif not outcome and streak_correct == 0:
            streak_incorrect += 1
        else:
            break
    snapshot.attempts_count = len(attempts)
    snapshot.correct_count = sum(outcomes)
    snapshot.rolling_accuracy_5 = _accuracy(attempts[-5:])
    snapshot.rolling_accuracy_10 = _accuracy(attempts[-10:])
    snapshot.mean_response_time_ms = sum(times) / len(times) if times else 0.0
    snapshot.recent_response_time_ms = float(last.response_time_ms if last else 0.0)
    snapshot.streak_correct = streak_correct
    snapshot.streak_incorrect = streak_incorrect
    snapshot.days_since_last_attempt = (
        max(0.0, (utcnow_naive() - last.created_at).total_seconds() / 86400.0) if last else 30.0
    )
    snapshot.mastery_estimate = (sum(outcomes) + 1.5) / (len(outcomes) + 3.0)
    snapshot.updated_at = utcnow_naive()
    db.flush()
    return snapshot


def refresh_all_snapshots(db: Session, user_id: int | None = None) -> int:
    query = db.query(models.Attempt.user_id, models.Task.skill_id).join(models.Task).distinct()
    if user_id is not None:
        query = query.filter(models.Attempt.user_id == user_id)
    pairs = query.all()
    for uid, skill_id in pairs:
        refresh_user_skill_snapshot(db, uid, skill_id)
    db.commit()
    return len(pairs)
