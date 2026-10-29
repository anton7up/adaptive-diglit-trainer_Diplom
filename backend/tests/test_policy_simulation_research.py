from __future__ import annotations

from sqlalchemy.orm import joinedload

from app import models
from app.services.model_training import pipeline_for_spec
from app.services.model_experiments import experiment_specs
from app.services.feature_engineering import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from app.services.policy_simulation import run_policy_robustness, run_policy_simulation
from ml_fixtures import make_frame


def test_policy_simulation_exposes_mastery_primary_endpoint(db_session):
    frame = make_frame(rows=260, users=10, seed=99)
    spec = next(spec for spec in experiment_specs(42) if spec.name == "logreg_c0_5")
    estimator = pipeline_for_spec(spec)
    estimator.fit(frame[NUMERIC_FEATURES + CATEGORICAL_FEATURES], frame["target"].astype(int))
    tasks = (
        db_session.query(models.Task)
        .options(joinedload(models.Task.skill))
        .filter(models.Task.active.is_(True))
        .all()
    )
    report = run_policy_simulation(
        tasks,
        estimator,
        learners=10,
        steps=24,
        random_seed=7,
        mastery_threshold=0.65,
        minimum_skill_mastery=0.40,
    )
    endpoint = report["primary_endpoint"]
    assert "learning" in report["primary_outcome"]
    assert endpoint["adaptive_policy"] == "ml_adaptive"
    assert endpoint["comparator"] == "random"
    assert "tasks_saved" in endpoint
    assert "mastery_gain_advantage" in report["learning_gain_endpoint"]
    assert len(report["learning_trajectories"]) == 4
    assert {row["policy"] for row in report["policies"]} == {"random", "difficulty_match", "weak_skill", "ml_adaptive"}


def test_policy_robustness_reports_seed_level_results(db_session):
    frame = make_frame(rows=240, users=10, seed=123)
    spec = next(spec for spec in experiment_specs(42) if spec.name == "logreg_c0_5")
    estimator = pipeline_for_spec(spec)
    estimator.fit(frame[NUMERIC_FEATURES + CATEGORICAL_FEATURES], frame["target"].astype(int))
    tasks = (
        db_session.query(models.Task)
        .options(joinedload(models.Task.skill))
        .filter(models.Task.active.is_(True))
        .all()
    )
    report = run_policy_robustness(tasks, estimator, seeds=(3, 5), learners=10, steps=20, mastery_threshold=0.65, minimum_skill_mastery=0.40)
    assert len(report["runs"]) == 2
    assert report["summary"]["positive_time_to_mastery_runs"] <= 2
    assert report["summary"]["positive_mastery_gain_runs"] <= 2
