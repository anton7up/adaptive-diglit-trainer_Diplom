from __future__ import annotations

from app import models
from app.services.adaptive_engine import _selection_components, _selection_score, rank_tasks
from app.services.seed import seed_synthetic_history


def test_selection_components_are_bounded_except_penalty():
    components = _selection_components(
        0.67,
        skill_accuracy=0.45,
        exposure_count=2,
        recently_seen=True,
        cognitive_level=2,
        target_probability=0.67,
    )
    assert components["learning_zone"] == 1.0
    assert 0.0 <= components["skill_weakness"] <= 1.0
    assert 0.0 < components["novelty"] <= 1.0
    assert 0.0 <= components["uncertainty"] <= 1.0
    assert 0.0 <= components["cognitive_fit"] <= 1.0
    assert components["recency_penalty"] == -0.65


def test_selection_score_rewards_learning_zone():
    near = _selection_components(0.67, skill_accuracy=0.5, exposure_count=0, recently_seen=False, cognitive_level=2, target_probability=0.67)
    far = _selection_components(0.98, skill_accuracy=0.5, exposure_count=0, recently_seen=False, cognitive_level=2, target_probability=0.67)
    assert _selection_score(near) > _selection_score(far)


def test_selection_score_penalizes_recent_task():
    fresh = _selection_components(0.67, skill_accuracy=0.5, exposure_count=1, recently_seen=False, cognitive_level=2, target_probability=0.67)
    recent = _selection_components(0.67, skill_accuracy=0.5, exposure_count=1, recently_seen=True, cognitive_level=2, target_probability=0.67)
    assert _selection_score(fresh) - _selection_score(recent) == 0.65


def test_rank_tasks_is_non_mutating_for_prediction_audit(db_session):
    student = db_session.query(models.User).filter_by(username="student_demo").one()
    tasks = db_session.query(models.Task).limit(8).all()
    before = db_session.query(models.PredictionAudit).count()
    ranked = rank_tasks(db_session, student.id, tasks, limit=4)
    after = db_session.query(models.PredictionAudit).count()
    assert len(ranked) == 4
    assert before == after
    assert all(ranked[index].score >= ranked[index + 1].score for index in range(len(ranked) - 1))


def test_rank_tasks_exposes_model_probability_and_components(db_session):
    student = db_session.query(models.User).filter_by(username="student_demo").one()
    tasks = db_session.query(models.Task).limit(10).all()
    ranked = rank_tasks(db_session, student.id, tasks, limit=3)
    for row in ranked:
        assert 0.0 < row.prediction.probability < 1.0
        assert row.prediction.model_version
        assert "learning_zone" in row.components
        assert "skill_weakness" in row.components
        assert "novelty" in row.components


def test_synthetic_history_changes_personalization_inputs(db_session):
    created = seed_synthetic_history(db_session, users=3, attempts_per_user=12)
    assert created == 36
    user = db_session.query(models.User).filter(models.User.username.like("synthetic_%")).first()
    tasks = db_session.query(models.Task).limit(12).all()
    ranked = rank_tasks(db_session, user.id, tasks, limit=5)
    assert len(ranked) == 5
    assert all(row.prediction.features["user_attempts_total"] == 12 for row in ranked)
    assert any(row.prediction.features["skill_attempts"] > 0 for row in ranked)
