from __future__ import annotations

from datetime import datetime, timedelta

from app.timeutils import utcnow_naive
from app import models
from app.services.snapshot_service import refresh_all_snapshots, refresh_user_skill_snapshot


def add_attempt(db, user, task, correct, offset, time_ms):
    db.add(models.Attempt(
        user_id=user.id,
        task_id=task.id,
        selected_option=task.correct_option if correct else ("D" if task.correct_option != "D" else "C"),
        is_correct=correct,
        response_time_ms=time_ms,
        created_at=utcnow_naive() + timedelta(minutes=offset),
    ))
    db.commit()


def test_snapshot_rolling_metrics(db_session):
    user = db_session.query(models.User).filter_by(username="student_demo").one()
    skill = db_session.query(models.Skill).first()
    tasks = db_session.query(models.Task).filter_by(skill_id=skill.id).limit(3).all()
    outcomes = [True, True, False, True, False, True]
    for index, outcome in enumerate(outcomes):
        add_attempt(db_session, user, tasks[index % len(tasks)], outcome, index, 1000 + index * 100)
    snapshot = refresh_user_skill_snapshot(db_session, user.id, skill.id)
    assert snapshot.attempts_count == 6
    assert snapshot.correct_count == 4
    assert abs(snapshot.rolling_accuracy_5 - 0.6) < 1e-9
    assert abs(snapshot.mastery_estimate - 5.5 / 9.0) < 1e-9


def test_snapshot_correct_streak(db_session):
    user = db_session.query(models.User).filter_by(username="student_demo").one()
    task = db_session.query(models.Task).first()
    for index, outcome in enumerate([False, True, True, True]):
        add_attempt(db_session, user, task, outcome, index, 1000)
    snapshot = refresh_user_skill_snapshot(db_session, user.id, task.skill_id)
    assert snapshot.streak_correct == 3
    assert snapshot.streak_incorrect == 0


def test_snapshot_incorrect_streak(db_session):
    user = db_session.query(models.User).filter_by(username="student_demo").one()
    task = db_session.query(models.Task).first()
    for index, outcome in enumerate([True, False, False]):
        add_attempt(db_session, user, task, outcome, index, 1000)
    snapshot = refresh_user_skill_snapshot(db_session, user.id, task.skill_id)
    assert snapshot.streak_correct == 0
    assert snapshot.streak_incorrect == 2


def test_refresh_all_snapshots(db_session):
    user = db_session.query(models.User).filter_by(username="student_demo").one()
    skills = db_session.query(models.Skill).limit(2).all()
    for index, skill in enumerate(skills):
        task = db_session.query(models.Task).filter_by(skill_id=skill.id).first()
        add_attempt(db_session, user, task, True, index, 1200)
    count = refresh_all_snapshots(db_session, user.id)
    assert count == 2
    assert db_session.query(models.UserSkillSnapshot).filter_by(user_id=user.id).count() == 2
