from __future__ import annotations

from app import models
from app.services.seed import seed_catalog, seed_synthetic_history


def test_seed_catalog_is_idempotent(db_session):
    result = seed_catalog(db_session)
    assert result == {"skills": 0, "tasks": 0, "users": 0, "exams": 0}
    assert db_session.query(models.Skill).count() == 6
    assert db_session.query(models.Task).count() == 120


def test_synthetic_history_is_explicitly_marked(db_session):
    created = seed_synthetic_history(db_session, users=5, attempts_per_user=10)
    assert created == 50
    synthetic = db_session.query(models.User).filter(models.User.username.like("synthetic_%")).all()
    assert len(synthetic) == 5
    assert all(not user.is_active for user in synthetic)
    attempts = db_session.query(models.Attempt).join(models.User).filter(models.User.username.like("synthetic_%")).all()
    assert all(item.client_context["source"] == "synthetic_bootstrap_v2" for item in attempts)


def test_synthetic_history_is_idempotent(db_session):
    assert seed_synthetic_history(db_session, users=5, attempts_per_user=10) == 50
    assert seed_synthetic_history(db_session, users=5, attempts_per_user=10) == 0
