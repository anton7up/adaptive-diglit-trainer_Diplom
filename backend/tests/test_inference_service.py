from __future__ import annotations

import numpy as np
from sqlalchemy.orm import joinedload

from app import models
from app.services.feature_engineering import build_training_frame
from app.services.inference import _fallback_probabilities, predict_tasks
from app.services.model_training import pipeline_for_spec
from app.services.seed import seed_synthetic_history
from ml_fixtures import fast_spec


def test_predict_tasks_uses_cold_start_when_registry_empty(db_session):
    student = db_session.query(models.User).filter_by(username="student_demo").one()
    tasks = db_session.query(models.Task).options(joinedload(models.Task.skill)).limit(6).all()
    predictions = predict_tasks(db_session, student.id, tasks)
    assert len(predictions) == 6
    assert all(row.model_version == "cold-start-baseline" for row in predictions)
    assert all(0.0 < row.probability < 1.0 for row in predictions)


def test_predict_tasks_accepts_explicit_estimator(db_session):
    seed_synthetic_history(db_session, users=5, attempts_per_user=14)
    attempts = (
        db_session.query(models.Attempt)
        .options(joinedload(models.Attempt.task).joinedload(models.Task.skill))
        .order_by(models.Attempt.created_at.asc(), models.Attempt.id.asc())
        .all()
    )
    frame = build_training_frame(attempts)
    estimator = pipeline_for_spec(fast_spec()).fit(frame, frame["target"].astype(int))
    user = db_session.query(models.User).filter(models.User.username.like("synthetic_%")).first()
    tasks = db_session.query(models.Task).options(joinedload(models.Task.skill)).limit(7).all()
    predictions = predict_tasks(db_session, user.id, tasks, estimator=estimator, model_version="pytest-model")
    assert len(predictions) == 7
    assert all(row.model_version == "pytest-model" for row in predictions)
    assert len({round(row.probability, 5) for row in predictions}) > 1


def test_fallback_probability_decreases_when_difficulty_increases():
    import pandas as pd
    frame = pd.DataFrame({
        "skill_accuracy": [0.65, 0.65],
        "user_accuracy_total": [0.65, 0.65],
        "task_difficulty": [0.25, 0.85],
    })
    probabilities = _fallback_probabilities(frame)
    assert probabilities[0] > probabilities[1]
    assert np.all((probabilities > 0) & (probabilities < 1))
