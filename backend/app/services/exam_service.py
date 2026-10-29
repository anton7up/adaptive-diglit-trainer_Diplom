from __future__ import annotations

from datetime import datetime, timedelta

from sqlalchemy.orm import Session, joinedload

from .. import models
from ..timeutils import utcnow_naive


def start_exam(db: Session, exam: models.Exam, user: models.User) -> models.ExamSession:
    active = (
        db.query(models.ExamSession)
        .filter_by(exam_id=exam.id, user_id=user.id, status="in_progress")
        .order_by(models.ExamSession.started_at.desc())
        .first()
    )
    if active is not None:
        if utcnow_naive() <= active.started_at + timedelta(minutes=exam.duration_minutes):
            return active
        active.status = "expired"
        active.submitted_at = utcnow_naive()
    session = models.ExamSession(exam_id=exam.id, user_id=user.id)
    db.add(session)
    db.commit()
    db.refresh(session)
    return session


def exam_payload(exam: models.Exam, session: models.ExamSession) -> dict:
    links = sorted(exam.tasks, key=lambda item: item.position)
    return {
        "session_id": session.id,
        "exam_id": exam.id,
        "title": exam.title,
        "duration_minutes": exam.duration_minutes,
        "started_at": session.started_at,
        "tasks": [
            {
                "id": link.task.id,
                "position": link.position,
                "question": link.task.question,
                "options": {
                    "A": link.task.option_a,
                    "B": link.task.option_b,
                    "C": link.task.option_c,
                    "D": link.task.option_d,
                },
                "points": link.points,
            }
            for link in links
        ],
    }


def answer_exam_task(db: Session, session: models.ExamSession, task_id: int, selected: str, response_time_ms: int) -> models.ExamAnswer:
    if session.status != "in_progress":
        raise ValueError("exam session is not active")
    if utcnow_naive() > session.started_at + timedelta(minutes=session.exam.duration_minutes):
        session.status = "expired"
        session.submitted_at = utcnow_naive()
        db.commit()
        raise ValueError("exam time limit exceeded")
    link = next((item for item in session.exam.tasks if item.task_id == task_id), None)
    if link is None:
        raise ValueError("task does not belong to exam")
    existing = db.query(models.ExamAnswer).filter_by(session_id=session.id, task_id=task_id).first()
    if existing is not None:
        raise ValueError("answer for this task is already recorded")
    correct = selected.upper() == link.task.correct_option
    answer = models.ExamAnswer(
        session_id=session.id,
        task_id=task_id,
        selected_option=selected.upper(),
        is_correct=correct,
        points_awarded=link.points if correct else 0.0,
        response_time_ms=response_time_ms,
    )
    db.add(answer)
    db.commit()
    db.refresh(answer)
    return answer


def submit_exam(db: Session, session: models.ExamSession) -> dict:
    if session.status != "in_progress":
        if session.submitted_at is None:
            raise ValueError("exam session cannot be submitted")
    links = session.exam.tasks
    total_points = sum(item.points for item in links)
    awarded = sum(item.points_awarded for item in session.answers)
    score = awarded / total_points if total_points else 0.0
    session.score = score
    session.passed = score >= session.exam.passing_score
    session.status = "submitted"
    session.submitted_at = utcnow_naive()
    db.commit()
    return {
        "session_id": session.id,
        "score": round(score, 4),
        "passed": bool(session.passed),
        "correct_count": sum(int(item.is_correct) for item in session.answers),
        "total_count": len(links),
        "submitted_at": session.submitted_at,
    }
