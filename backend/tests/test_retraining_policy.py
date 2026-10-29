from __future__ import annotations

from datetime import datetime, timedelta

from app.timeutils import utcnow_naive
from app import models
from app.services.retraining_policy import evaluate_retraining_need


def test_no_active_model_requests_training(db_session):
    decision = evaluate_retraining_need(db_session)
    assert decision.should_retrain
    assert "no_active_model" in decision.reasons


def test_fresh_active_model_with_no_attempts_does_not_retrain(db_session):
    row = models.MLModelVersion(
        version="v1",
        model_name="logreg",
        artifact_path="/tmp/missing.joblib",
        metadata_path="/tmp/missing.json",
        training_run_id="run1",
        is_active=True,
        metrics={},
        feature_names=[],
        trained_at=utcnow_naive(),
    )
    db_session.add(row); db_session.commit()
    decision = evaluate_retraining_need(db_session, min_new_attempts=10, max_age_hours=100)
    assert not decision.should_retrain


def test_old_model_requests_retraining(db_session):
    row = models.MLModelVersion(
        version="v-old",
        model_name="logreg",
        artifact_path="/tmp/a",
        metadata_path="/tmp/b",
        training_run_id="run-old",
        is_active=True,
        metrics={},
        feature_names=[],
        trained_at=utcnow_naive() - timedelta(days=40),
    )
    db_session.add(row); db_session.commit()
    decision = evaluate_retraining_need(db_session, max_age_hours=24 * 14)
    assert decision.should_retrain
    assert "model_age_limit" in decision.reasons
