from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from app.services.feature_engineering import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from app.services.model_experiments import ExperimentSpec
from app.services.model_training import pipeline_for_spec


def make_frame(rows: int = 320, users: int = 12, seed: int = 123) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    user_id = np.arange(rows) % users
    task_id = np.arange(rows) % 36
    skill_index = task_id % 6
    difficulty = np.clip(0.12 + (task_id % 10) * 0.075, 0.08, 0.92)
    base_ability = 0.26 + 0.55 * (user_id / max(users - 1, 1))
    practice = np.minimum(np.arange(rows) // users / 30.0, 0.16)
    ability = np.clip(base_ability + practice + rng.normal(0, 0.04, rows), 0.04, 0.96)
    probability = 1.0 / (1.0 + np.exp(-5.0 * (ability - difficulty)))
    target = (rng.random(rows) < probability).astype(int)
    skill_accuracy = np.clip(ability + rng.normal(0, 0.07, rows), 0.02, 0.98)

    data = {name: rng.normal(size=rows) for name in NUMERIC_FEATURES}
    data.update({
        "user_attempts_total": np.arange(rows) // users,
        "user_accuracy_total": np.clip(ability + rng.normal(0, 0.06, rows), 0.01, 0.99),
        "user_accuracy_last5": np.clip(ability + rng.normal(0, 0.09, rows), 0.01, 0.99),
        "user_accuracy_last10": np.clip(ability + rng.normal(0, 0.07, rows), 0.01, 0.99),
        "user_attempts_24h": rng.integers(0, 8, rows),
        "user_attempts_7d": rng.integers(1, 30, rows),
        "skill_attempts": np.arange(rows) % 22,
        "skill_accuracy": skill_accuracy,
        "skill_accuracy_last5": np.clip(skill_accuracy + rng.normal(0, 0.07, rows), 0.01, 0.99),
        "skill_accuracy_last10": np.clip(skill_accuracy + rng.normal(0, 0.05, rows), 0.01, 0.99),
        "skill_attempts_7d": rng.integers(0, 12, rows),
        "task_attempts": np.arange(rows) % 5,
        "task_accuracy": np.clip(0.50 + rng.normal(0, 0.16, rows), 0.01, 0.99),
        "task_exposure_share": rng.uniform(0.0, 0.20, rows),
        "skill_exposure_share": rng.uniform(0.05, 0.35, rows),
        "seconds_since_last_attempt": rng.uniform(60, 4 * 86400, rows),
        "seconds_since_last_skill_attempt": rng.uniform(60, 12 * 86400, rows),
        "seconds_since_last_task_attempt": rng.uniform(60, 30 * 86400, rows),
        "response_time_mean_sec": rng.uniform(15, 70, rows),
        "response_time_std_sec": rng.uniform(0, 20, rows),
        "response_time_last_sec": rng.uniform(10, 100, rows),
        "skill_response_time_mean_sec": rng.uniform(15, 75, rows),
        "skill_response_time_last_sec": rng.uniform(10, 100, rows),
        "correct_streak": rng.integers(0, 6, rows),
        "incorrect_streak": rng.integers(0, 5, rows),
        "skill_correct_streak": rng.integers(0, 5, rows),
        "skill_incorrect_streak": rng.integers(0, 4, rows),
        "task_difficulty": difficulty,
        "task_discrimination": rng.uniform(0.7, 1.4, rows),
        "task_cognitive_level": rng.integers(1, 4, rows),
        "task_estimated_time_sec": rng.choice([30, 40, 50, 60, 70], rows),
        "difficulty_gap": difficulty - skill_accuracy,
        "speed_ratio": rng.uniform(0.4, 2.0, rows),
        "practice_density_7d": rng.uniform(0.1, 5.0, rows),
        "local_hour_sin": np.sin(2 * np.pi * (np.arange(rows) % 24) / 24),
        "local_hour_cos": np.cos(2 * np.pi * (np.arange(rows) % 24) / 24),
        "is_first_skill_attempt": (np.arange(rows) % 22 == 0).astype(int),
        "is_first_task_attempt": (np.arange(rows) % 5 == 0).astype(int),
        "skill_code": np.asarray([f"S{value}" for value in skill_index]),
        "target": target,
        "user_id": user_id,
        "task_id": task_id,
        "attempt_id": np.arange(1, rows + 1),
        "created_at": [datetime(2026, 1, 1) + timedelta(hours=index * 2) for index in range(rows)],
    })
    return pd.DataFrame(data)


def fast_spec(seed: int = 42) -> ExperimentSpec:
    return ExperimentSpec(
        name="logreg_test",
        family="logistic_regression",
        scale_numeric=True,
        estimator=LogisticRegression(C=1.0, max_iter=800, class_weight="balanced", random_state=seed),
        parameters={"C": 1.0, "max_iter": 800, "class_weight": "balanced"},
    )


def fitted_estimator(rows: int = 320, users: int = 12):
    frame = make_frame(rows, users)
    estimator = pipeline_for_spec(fast_spec())
    estimator.fit(frame[NUMERIC_FEATURES + CATEGORICAL_FEATURES], frame["target"].astype(int))
    return estimator, frame
