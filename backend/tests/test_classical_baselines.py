from __future__ import annotations

from app.services.classical_baselines import benchmark_classical_baselines
from ml_fixtures import make_frame


def test_classical_baseline_benchmark_includes_required_families():
    frame = make_frame(rows=360, users=12, seed=321)
    train_users = list(range(0, 9))
    validation_users = list(range(9, 12))
    metadata = {
        "best_model": "logreg_c0_5",
        "best_family": "logistic_regression",
        "best_parameters": {"C": 1.0},
        "split": {
            "strategy": "group_holdout",
            "train_users": train_users,
            "validation_users": validation_users,
        },
    }
    report = benchmark_classical_baselines(frame, metadata)
    names = {row["name"] for row in report["models"]}
    assert {"ML adaptive predictor", "Elo", "IRT-2PL online", "BKT", "PFA"}.issubset(names)
    assert report["split"]["user_overlap"] == 0
    assert len(report["ml_vs_classical"]) == 4
    assert all("p_value_holm_two_sided" in row for row in report["ml_vs_classical"])
    assert all(0.0 <= row["metrics"]["brier"] <= 1.0 for row in report["models"])
