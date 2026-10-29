from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from app.services.feature_engineering import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from app.services.model_experiments import experiment_specs
from app.services.model_training import composite_metric, probability_metrics, train_and_compare
from tests.ml_fixtures import fast_spec


def synthetic_frame(rows=360, users=12):
    rng = np.random.default_rng(42)
    user_id = np.arange(rows) % users
    difficulty = rng.uniform(0.1, 0.9, rows)
    ability = 0.25 + (user_id / max(users - 1, 1)) * 0.6
    skill_accuracy = np.clip(ability + rng.normal(0, 0.08, rows), 0.02, 0.98)
    logits = 5.0 * (ability - difficulty)
    probability = 1 / (1 + np.exp(-logits))
    target = (rng.random(rows) < probability).astype(int)

    data = {name: rng.normal(size=rows) for name in NUMERIC_FEATURES}
    data.update({
        "user_attempts_total": np.arange(rows) % 40,
        "user_accuracy_total": np.clip(ability + rng.normal(0, 0.06, rows), 0.01, 0.99),
        "skill_attempts": np.arange(rows) % 18,
        "skill_accuracy": skill_accuracy,
        "skill_accuracy_last5": np.clip(skill_accuracy + rng.normal(0, 0.07, rows), 0.01, 0.99),
        "skill_accuracy_last10": np.clip(skill_accuracy + rng.normal(0, 0.05, rows), 0.01, 0.99),
        "task_attempts": np.arange(rows) % 4,
        "task_accuracy": np.clip(0.5 + rng.normal(0, 0.15, rows), 0.01, 0.99),
        "seconds_since_last_attempt": rng.uniform(100, 100000, rows),
        "seconds_since_last_skill_attempt": rng.uniform(100, 300000, rows),
        "response_time_mean_sec": rng.uniform(8, 80, rows),
        "response_time_last_sec": rng.uniform(5, 100, rows),
        "correct_streak": rng.integers(0, 5, rows),
        "incorrect_streak": rng.integers(0, 4, rows),
        "task_difficulty": difficulty,
        "task_discrimination": rng.uniform(0.7, 1.4, rows),
        "task_cognitive_level": rng.integers(1, 4, rows),
        "task_estimated_time_sec": rng.choice([30, 40, 50, 60, 70], rows),
        "difficulty_gap": difficulty - skill_accuracy,
        "is_first_skill_attempt": (np.arange(rows) % 18 == 0).astype(int),
        "is_first_task_attempt": (np.arange(rows) % 4 == 0).astype(int),
        "skill_code": np.array([f"S{i % 6}" for i in range(rows)]),
        "target": target,
        "user_id": user_id,
        "task_id": np.arange(rows) % 30,
        "attempt_id": np.arange(1, rows + 1),
        "created_at": [datetime(2026, 1, 1) + timedelta(hours=i) for i in range(rows)],
    })
    return pd.DataFrame(data)


def test_experiment_space_has_multiple_variants_per_family():
    specs = experiment_specs(42)
    assert len(specs) == 8
    families = {}
    for spec in specs:
        families.setdefault(spec.family, 0)
        families[spec.family] += 1
    assert families == {
        "logistic_regression": 2,
        "random_forest": 2,
        "extra_trees": 2,
        "hist_gradient_boosting": 2,
    }


def test_probability_metrics_have_expected_keys():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.4, 0.6, 0.9])
    result = probability_metrics(y, p)
    assert set(result) == {"accuracy", "precision", "recall", "f1", "log_loss", "brier", "roc_auc", "pr_auc"}
    assert result["roc_auc"] == 1.0
    assert result["brier"] < 0.1


def test_composite_metric_rewards_better_calibration():
    common = {"roc_auc": 0.75, "pr_auc": 0.76, "f1": 0.70, "log_loss": 0.60, "brier": 0.18}
    worse = {**common, "brier": 0.31}
    assert composite_metric(common) > composite_metric(worse)


def test_train_and_compare_persists_artifacts(tmp_path):
    frame = synthetic_frame(240, 10)
    result = train_and_compare(frame, artifact_dir=tmp_path, specs=[fast_spec()])
    assert result["model_name"]
    assert result["version"]
    assert (tmp_path / f"{result['version']}.joblib").exists()
    assert (tmp_path / f"{result['version']}.json").exists()
    assert result["metadata"]["experiment_count"] == 1
    assert len(result["metadata"]["candidates"]) == 1
    assert result["metadata"]["dataset_rows"] == 240


def test_selected_candidate_has_highest_composite(tmp_path):
    result = train_and_compare(synthetic_frame(180, 8), artifact_dir=tmp_path, specs=[fast_spec()])
    candidates = result["metadata"]["candidates"]
    scores = [item["composite_score"] for item in candidates]
    assert scores == sorted(scores, reverse=True)
    assert result["model_name"] == candidates[0]["name"]


def test_trained_estimator_produces_probability(tmp_path):
    frame = synthetic_frame(160, 8)
    result = train_and_compare(frame, artifact_dir=tmp_path, specs=[fast_spec()])
    estimator = result["estimator"]
    p = estimator.predict_proba(frame[NUMERIC_FEATURES + CATEGORICAL_FEATURES].head(5))[:, 1]
    assert len(p) == 5
    assert np.all((p >= 0) & (p <= 1))


import pytest


@pytest.mark.slow
def test_full_experiment_space_trains_all_candidates(tmp_path):
    frame = synthetic_frame(180, 12)
    result = train_and_compare(frame, artifact_dir=tmp_path)
    candidates = result["metadata"]["candidates"]
    assert result["metadata"]["experiment_count"] == 8
    assert len(candidates) == 8
    assert {item["family"] for item in candidates} == {
        "logistic_regression",
        "random_forest",
        "extra_trees",
        "hist_gradient_boosting",
    }
    assert result["metadata"]["split"]["user_overlap"] == []
