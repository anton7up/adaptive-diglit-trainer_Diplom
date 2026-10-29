from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd

from app.timeutils import utcnow_naive
from app import models
from app.services.feature_engineering import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    HistoryAccumulator,
    build_candidate_frame,
    build_training_frame,
    validate_feature_frame,
)


def _make_attempt(db, user_id, task, correct, when, response=10000):
    attempt = models.Attempt(
        user_id=user_id,
        task_id=task.id,
        selected_option=task.correct_option if correct else "D" if task.correct_option != "D" else "C",
        is_correct=correct,
        response_time_ms=response,
        created_at=when,
    )
    db.add(attempt)
    db.commit()
    db.refresh(attempt)
    return attempt


def test_training_frame_has_expected_columns(db_session):
    user = db_session.query(models.User).filter_by(username="student_demo").one()
    task = db_session.query(models.Task).first()
    _make_attempt(db_session, user.id, task, True, utcnow_naive() - timedelta(hours=2))
    _make_attempt(db_session, user.id, task, False, utcnow_naive() - timedelta(hours=1))
    attempts = db_session.query(models.Attempt).order_by(models.Attempt.created_at).all()
    frame = build_training_frame(attempts)
    for column in NUMERIC_FEATURES + CATEGORICAL_FEATURES + ["target", "user_id", "task_id"]:
        assert column in frame.columns
    validate_feature_frame(frame, require_target=True)


def test_first_observation_does_not_leak_target(db_session):
    user = db_session.query(models.User).filter_by(username="student_demo").one()
    task = db_session.query(models.Task).first()
    attempt = _make_attempt(db_session, user.id, task, True, utcnow_naive())
    frame = build_training_frame([attempt])
    row = frame.iloc[0]
    assert row["target"] == 1
    assert row["user_attempts_total"] == 0
    assert row["skill_attempts"] == 0
    assert row["task_attempts"] == 0
    assert row["skill_accuracy"] == 0.5


def test_second_observation_uses_only_previous_history(db_session):
    user = db_session.query(models.User).filter_by(username="student_demo").one()
    task = db_session.query(models.Task).first()
    now = utcnow_naive()
    first = _make_attempt(db_session, user.id, task, True, now - timedelta(hours=1))
    second = _make_attempt(db_session, user.id, task, False, now)
    frame = build_training_frame([first, second])
    row = frame.iloc[1]
    assert row["user_attempts_total"] == 1
    assert row["skill_attempts"] == 1
    assert row["task_attempts"] == 1
    assert row["correct_streak"] == 1
    assert row["target"] == 0


def test_history_accumulator_updates_streaks(db_session):
    user = db_session.query(models.User).filter_by(username="student_demo").one()
    task = db_session.query(models.Task).first()
    now = utcnow_naive()
    acc = HistoryAccumulator()
    a1 = _make_attempt(db_session, user.id, task, True, now - timedelta(minutes=2))
    a2 = _make_attempt(db_session, user.id, task, True, now - timedelta(minutes=1))
    acc.consume(a1); acc.consume(a2)
    features = acc.features(user.id, task, task.skill.code, now)
    assert features["correct_streak"] == 2
    assert features["incorrect_streak"] == 0


def test_candidate_frame_contains_one_row_per_task(db_session):
    user = db_session.query(models.User).filter_by(username="student_demo").one()
    tasks = db_session.query(models.Task).limit(7).all()
    frame = build_candidate_frame(db_session, user.id, tasks)
    assert len(frame) == 7
    assert list(frame.columns)


def test_response_time_features_are_seconds(db_session):
    user = db_session.query(models.User).filter_by(username="student_demo").one()
    task = db_session.query(models.Task).first()
    a1 = _make_attempt(db_session, user.id, task, True, utcnow_naive() - timedelta(minutes=1), response=12500)
    frame = build_training_frame([a1])
    assert frame.iloc[0]["response_time_mean_sec"] == 0.0
    candidate = build_candidate_frame(db_session, user.id, [task])
    assert abs(candidate.iloc[0]["response_time_mean_sec"] - 12.5) < 1e-9


def test_skill_code_is_categorical_string(db_session):
    user = db_session.query(models.User).filter_by(username="student_demo").one()
    task = db_session.query(models.Task).first()
    frame = build_candidate_frame(db_session, user.id, [task])
    assert isinstance(frame.iloc[0]["skill_code"], str)


def test_numeric_features_are_finite(db_session):
    user = db_session.query(models.User).filter_by(username="student_demo").one()
    tasks = db_session.query(models.Task).limit(10).all()
    frame = build_candidate_frame(db_session, user.id, tasks)
    validate_feature_frame(frame)
    assert not frame[NUMERIC_FEATURES].isna().any().any()
