from __future__ import annotations

from sqlalchemy.orm import joinedload

from app import models
from app.services.feature_engineering import build_training_frame
from app.services.model_training import pipeline_for_spec
from app.services.personalization_diagnostics import (
    challenge_zone_diagnostics,
    personalization_report,
    probability_surface,
    ranking_diversity,
)
from app.services.seed import seed_synthetic_history
from ml_fixtures import fast_spec


def prepared_context(db_session, users=6, attempts=16):
    created = seed_synthetic_history(db_session, users=users, attempts_per_user=attempts)
    assert created == users * attempts
    rows = (
        db_session.query(models.Attempt)
        .options(joinedload(models.Attempt.task).joinedload(models.Task.skill))
        .order_by(models.Attempt.created_at.asc(), models.Attempt.id.asc())
        .all()
    )
    frame = build_training_frame(rows)
    estimator = pipeline_for_spec(fast_spec())
    estimator.fit(frame, frame["target"].astype(int))
    synthetic_users = (
        db_session.query(models.User)
        .filter(models.User.username.like("synthetic_%"))
        .order_by(models.User.id.asc())
        .all()
    )
    tasks = (
        db_session.query(models.Task)
        .options(joinedload(models.Task.skill))
        .filter(models.Task.active.is_(True))
        .order_by(models.Task.id.asc())
        .all()
    )
    return estimator, synthetic_users, tasks


def test_probability_surface_contains_user_and_task_variation(db_session):
    estimator, users, tasks = prepared_context(db_session)
    report = probability_surface(db_session, estimator, users, tasks[:20])
    assert report["predictions"] == len(users) * 20
    assert report["probability"]["count"] == len(users) * 20
    assert report["within_user_probability_spread"]["max"] > 0
    assert report["task_mean_probability"]["count"] == 20


def test_ranking_diversity_returns_top_choice_distribution(db_session):
    estimator, users, tasks = prepared_context(db_session)
    report = ranking_diversity(db_session, estimator, users, tasks[:24], top_k=4)
    assert report["users_ranked"] == len(users)
    assert 1 <= report["unique_first_choices"] <= 24
    assert 1 <= report["unique_top_k_tasks"] <= 24
    assert 0.0 <= report["top_k_task_entropy"] <= 1.0
    assert 0.0 <= report["pairwise_top_k_disagreement"] <= 1.0
    assert sum(row["count"] for row in report["skill_distribution"]) == len(users) * 4


def test_challenge_zone_diagnostics_has_valid_probability_distribution(db_session):
    estimator, users, tasks = prepared_context(db_session)
    report = challenge_zone_diagnostics(db_session, estimator, users, tasks[:30])
    assert report["selections"] == len(users)
    assert 0.0 <= report["productive_zone_share"] <= 1.0
    assert 0.0 <= report["selected_probability"]["min"] <= 1.0
    assert 0.0 <= report["selected_probability"]["max"] <= 1.0


def test_personalization_report_combines_diagnostics(db_session):
    estimator, _, _ = prepared_context(db_session, users=7, attempts=14)
    report = personalization_report(db_session, estimator, max_users=7, minimum_attempts=8, top_k=5)
    assert report["users"] == 7
    assert report["tasks"] == 120
    assert report["probability_surface"]["predictions"] == 7 * 120
    assert report["ranking_diversity"]["users_ranked"] == 7
    assert report["challenge_zone"]["selections"] == 7
    assert "user-dependent" in report["interpretation"]
