from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta

from sqlalchemy.orm import Session

from .. import models
from ..timeutils import utcnow_naive


def student_progress(db: Session, user: models.User) -> dict:
    attempts = (
        db.query(models.Attempt)
        .filter(models.Attempt.user_id == user.id)
        .order_by(models.Attempt.created_at.asc())
        .all()
    )
    skills = db.query(models.Skill).filter(models.Skill.active.is_(True)).order_by(models.Skill.sort_order).all()
    grouped: dict[int, list[models.Attempt]] = defaultdict(list)
    for attempt in attempts:
        grouped[attempt.task.skill_id].append(attempt)
    rows = []
    for skill in skills:
        values = grouped.get(skill.id, [])
        outcomes = [int(item.is_correct) for item in values]
        times = [item.response_time_ms for item in values if item.response_time_ms > 0]
        n = len(outcomes)
        correct = sum(outcomes)
        rolling5 = sum(outcomes[-5:]) / len(outcomes[-5:]) if outcomes else 0.5
        rolling10 = sum(outcomes[-10:]) / len(outcomes[-10:]) if outcomes else 0.5
        accuracy = correct / n if n else 0.0
        mastery = (correct + 1.5) / (n + 3.0)
        rows.append({
            "skill_id": skill.id,
            "skill_code": skill.code,
            "skill_name": skill.name,
            "attempts_count": n,
            "correct_count": correct,
            "accuracy": round(accuracy, 4),
            "rolling_accuracy_5": round(rolling5, 4),
            "rolling_accuracy_10": round(rolling10, 4),
            "mean_response_time_ms": round(sum(times) / len(times), 1) if times else 0.0,
            "mastery_estimate": round(mastery, 4),
        })
    total = len(attempts)
    total_correct = sum(int(item.is_correct) for item in attempts)
    return {
        "user_id": user.id,
        "username": user.username,
        "attempts_total": total,
        "correct_total": total_correct,
        "overall_accuracy": round(total_correct / total, 4) if total else 0.0,
        "rows": rows,
    }


def teacher_dashboard(db: Session) -> dict:
    students = db.query(models.User).filter(models.User.role == "student", models.User.is_active.is_(True)).all()
    attempts = db.query(models.Attempt).join(models.User).filter(models.User.is_active.is_(True)).all()
    cutoff = utcnow_naive() - timedelta(days=7)
    active_ids = {item.user_id for item in attempts if item.created_at >= cutoff}

    student_rows = []
    by_user: dict[int, list[models.Attempt]] = defaultdict(list)
    for attempt in attempts:
        by_user[attempt.user_id].append(attempt)
    for student in students:
        items = by_user.get(student.id, [])
        times = [a.response_time_ms for a in items if a.response_time_ms > 0]
        student_rows.append({
            "user_id": student.id,
            "username": student.username,
            "full_name": student.full_name,
            "attempts": len(items),
            "accuracy": round(sum(int(a.is_correct) for a in items) / len(items), 4) if items else 0.0,
            "avg_response_time_ms": round(sum(times) / len(times), 1) if times else 0.0,
            "last_activity_at": max((a.created_at for a in items), default=None),
        })

    skill_rows = []
    for skill in db.query(models.Skill).filter(models.Skill.active.is_(True)).order_by(models.Skill.sort_order).all():
        items = [a for a in attempts if a.task.skill_id == skill.id]
        calibrated = [a for a in items if a.predicted_probability is not None]
        actual = sum(int(a.is_correct) for a in items) / len(items) if items else 0.0
        avg_pred = sum(a.predicted_probability for a in calibrated) / len(calibrated) if calibrated else None
        skill_rows.append({
            "skill_id": skill.id,
            "code": skill.code,
            "name": skill.name,
            "attempts": len(items),
            "accuracy": round(actual, 4),
            "avg_predicted_probability": round(avg_pred, 4) if avg_pred is not None else None,
            "calibration_gap": round(abs(actual - avg_pred), 4) if avg_pred is not None else None,
        })

    daily: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for attempt in attempts:
        key = attempt.created_at.date().isoformat()
        daily[key][0] += 1
        daily[key][1] += int(attempt.is_correct)
    timeline = [
        {"date": key, "attempts": count, "correct": correct, "accuracy": round(correct / count, 4)}
        for key, (count, correct) in sorted(daily.items())[-30:]
    ]
    total = len(attempts)
    correct = sum(int(a.is_correct) for a in attempts)
    return {
        "students_total": len(students),
        "attempts_total": total,
        "active_students_7d": len(active_ids),
        "overall_accuracy": round(correct / total, 4) if total else 0.0,
        "students": student_rows,
        "skills": skill_rows,
        "timeline": timeline,
    }
