from __future__ import annotations

from app.services.validation_protocol import (
    build_validation_protocol,
    class_balance_check,
    entity_coverage_check,
    experiment_space_check,
    historical_count_consistency,
    metric_completeness_check,
    predictor_leakage_check,
    split_overlap_check,
    split_size_check,
    temporal_order_check,
)
from ml_fixtures import make_frame


def metadata(overlap=None, candidates=4):
    metric_set = {
        "accuracy": 0.7,
        "precision": 0.7,
        "recall": 0.7,
        "f1": 0.7,
        "roc_auc": 0.75,
        "pr_auc": 0.76,
        "log_loss": 0.58,
        "brier": 0.20,
    }
    families = ["logistic_regression", "random_forest", "extra_trees", "gradient_boosting"]
    return {
        "split": {
            "method": "group_holdout",
            "train_rows": 240,
            "validation_rows": 80,
            "user_overlap": [] if overlap is None else overlap,
        },
        "candidates": [
            {"name": f"m{index}", "family": families[index % 4], "validation": dict(metric_set)}
            for index in range(candidates)
        ],
    }


def test_predictor_leakage_names_are_clean():
    result = predictor_leakage_check()
    assert result["passed"] is True
    assert result["value"]["forbidden"] == []
    assert result["value"]["suspicious"] == []


def test_class_balance_passes_for_regular_training_frame():
    result = class_balance_check(make_frame(300, 12))
    assert result["passed"] is True
    assert 0.1 <= result["value"]["minority_rate"] <= 0.5


def test_class_balance_rejects_one_class_sample():
    frame = make_frame(100, 8)
    frame["target"] = 1
    assert class_balance_check(frame)["passed"] is False


def test_entity_coverage_counts_users_and_tasks():
    result = entity_coverage_check(make_frame(240, 12))
    assert result["passed"] is True
    assert result["value"]["users"] == 12
    assert result["value"]["tasks"] == 36


def test_temporal_order_passes_for_sorted_frame():
    assert temporal_order_check(make_frame(120, 8))["passed"] is True


def test_temporal_order_rejects_reverse_order():
    frame = make_frame(120, 8).iloc[::-1].reset_index(drop=True)
    result = temporal_order_check(frame)
    assert result["passed"] is False
    assert result["value"]["monotonic"] is False


def test_historical_counts_accepts_causal_counters():
    frame = make_frame(240, 12)
    # ml_fixtures interleaves users and increments this counter once per round.
    result = historical_count_consistency(frame)
    assert result["passed"] is True


def test_split_overlap_requires_zero_shared_users():
    assert split_overlap_check(metadata())["passed"] is True
    bad = split_overlap_check(metadata(overlap=[1, 2]))
    assert bad["passed"] is False
    assert bad["value"]["overlap"] == [1, 2]


def test_split_size_requires_nontrivial_holdout():
    good = metadata()
    assert split_size_check(good)["passed"] is True
    good["split"]["validation_rows"] = 5
    assert split_size_check(good)["passed"] is False


def test_experiment_space_requires_four_families():
    assert experiment_space_check(metadata())["passed"] is True
    assert experiment_space_check(metadata(candidates=3))["passed"] is False


def test_metric_completeness_detects_missing_probability_metric():
    good = metadata()
    assert metric_completeness_check(good)["passed"] is True
    del good["candidates"][0]["validation"]["brier"]
    result = metric_completeness_check(good)
    assert result["passed"] is False
    assert result["value"]["incomplete_candidates"] == ["m0"]


def test_full_validation_protocol_passes_consistent_inputs():
    report = build_validation_protocol(make_frame(320, 16), metadata())
    assert report["passed"] is True
    assert report["checks_passed"] == report["checks_total"]
    assert report["protocol"]["feature_count"] >= 35
    assert len(report["protocol"]["principles"]) >= 5
