from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .feature_engineering import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET_COLUMN


FORBIDDEN_PREDICTOR_NAMES = {
    TARGET_COLUMN,
    "is_correct",
    "correct_option",
    "selected_option",
    "answer",
    "label",
    "outcome",
}


def _check(name: str, passed: bool, value: Any, rationale: str) -> dict[str, Any]:
    return {
        "name": name,
        "passed": bool(passed),
        "value": value,
        "rationale": rationale,
    }


def predictor_leakage_check() -> dict[str, Any]:
    features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    forbidden = sorted(set(features) & FORBIDDEN_PREDICTOR_NAMES)
    suspicious = sorted(
        feature
        for feature in features
        if any(token in feature.lower() for token in ("target", "label", "outcome", "is_correct", "correct_option"))
    )
    passed = not forbidden and not suspicious
    return _check(
        "predictor_names_do_not_contain_target",
        passed,
        {"forbidden": forbidden, "suspicious": suspicious},
        "The response being predicted must never be part of the feature vector.",
    )


def class_balance_check(frame: pd.DataFrame, minimum_minority_rate: float = 0.10) -> dict[str, Any]:
    if TARGET_COLUMN not in frame or frame.empty:
        return _check("class_balance", False, None, "Training requires both target classes.")
    target = frame[TARGET_COLUMN].astype(int)
    positive_rate = float(target.mean())
    minority_rate = float(min(positive_rate, 1.0 - positive_rate))
    return _check(
        "class_balance",
        target.nunique() == 2 and minority_rate >= minimum_minority_rate,
        {"positive_rate": positive_rate, "minority_rate": minority_rate},
        "A severely imbalanced or one-class sample makes probability metrics unstable.",
    )


def entity_coverage_check(frame: pd.DataFrame, minimum_users: int = 5, minimum_tasks: int = 10) -> dict[str, Any]:
    users = int(frame["user_id"].nunique()) if "user_id" in frame else 0
    tasks = int(frame["task_id"].nunique()) if "task_id" in frame else 0
    return _check(
        "entity_coverage",
        users >= minimum_users and tasks >= minimum_tasks,
        {"users": users, "tasks": tasks},
        "Validation needs multiple users and tasks instead of repeated observations from one entity.",
    )


def temporal_order_check(frame: pd.DataFrame) -> dict[str, Any]:
    if "created_at" not in frame or frame.empty:
        return _check("temporal_order", False, None, "Attempt timestamps are required for causal feature construction and temporal backtesting.")
    timestamps = pd.to_datetime(frame["created_at"], errors="coerce")
    missing = int(timestamps.isna().sum())
    ordered = bool(timestamps.is_monotonic_increasing) if not missing else False
    return _check(
        "temporal_order",
        missing == 0 and ordered,
        {
            "missing_timestamps": missing,
            "monotonic": ordered,
            "start": timestamps.min().isoformat() if missing < len(timestamps) else None,
            "end": timestamps.max().isoformat() if missing < len(timestamps) else None,
        },
        "Rows must be processed in chronological order so every history feature is computed from the past only.",
    )


def historical_count_consistency(frame: pd.DataFrame) -> dict[str, Any]:
    required = {"user_id", "user_attempts_total", "skill_attempts", "task_attempts"}
    if not required.issubset(frame.columns):
        return _check("historical_counts", False, {"missing": sorted(required - set(frame.columns))}, "Pre-attempt counts should be available.")

    violations = 0
    first_rows = 0
    for _, group in frame.groupby("user_id", sort=False):
        group = group.sort_values("created_at", kind="stable") if "created_at" in group else group
        totals = pd.to_numeric(group["user_attempts_total"], errors="coerce").to_numpy(dtype=float)
        if len(totals):
            first_rows += 1
            if abs(float(totals[0])) > 1e-9:
                violations += 1
            if np.any(np.diff(totals) < -1e-9):
                violations += 1
    return _check(
        "historical_counts",
        violations == 0 and first_rows > 0,
        {"users_checked": first_rows, "violations": violations},
        "Pre-attempt counters should start from zero for a user's first observation and never move backwards.",
    )


def split_overlap_check(metadata: dict[str, Any]) -> dict[str, Any]:
    split = metadata.get("split", {}) if isinstance(metadata, dict) else {}
    overlap = split.get("user_overlap")
    if overlap is None:
        return _check("holdout_user_overlap", False, None, "Model metadata does not contain a split manifest.")
    return _check(
        "holdout_user_overlap",
        len(overlap) == 0,
        {"overlap": overlap, "method": split.get("method")},
        "For group holdout, the same learner must not occur in both training and validation subsets.",
    )


def split_size_check(metadata: dict[str, Any]) -> dict[str, Any]:
    split = metadata.get("split", {}) if isinstance(metadata, dict) else {}
    train_rows = int(split.get("train_rows", 0) or 0)
    validation_rows = int(split.get("validation_rows", 0) or 0)
    return _check(
        "holdout_size",
        train_rows >= 50 and validation_rows >= 20,
        {"train_rows": train_rows, "validation_rows": validation_rows},
        "Both training and holdout samples need enough observations for a meaningful engineering evaluation.",
    )


def experiment_space_check(metadata: dict[str, Any], expected_minimum: int = 4) -> dict[str, Any]:
    candidates = metadata.get("candidates", []) if isinstance(metadata, dict) else []
    families = sorted({str(row.get("family")) for row in candidates if row.get("family")})
    return _check(
        "model_comparison_space",
        len(candidates) >= expected_minimum and len(families) >= 4,
        {"candidates": len(candidates), "families": families},
        "The selected model should result from a comparison of materially different model families.",
    )


def metric_completeness_check(metadata: dict[str, Any]) -> dict[str, Any]:
    candidates = metadata.get("candidates", []) if isinstance(metadata, dict) else []
    expected = {"accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "log_loss", "brier"}
    incomplete: list[str] = []
    for candidate in candidates:
        metrics = candidate.get("validation", {})
        if not expected.issubset(metrics):
            incomplete.append(str(candidate.get("name", "unknown")))
    return _check(
        "probability_metric_coverage",
        bool(candidates) and not incomplete,
        {"required_metrics": sorted(expected), "incomplete_candidates": incomplete},
        "Probability models require discrimination, classification and calibration-sensitive metrics.",
    )


def build_validation_protocol(frame: pd.DataFrame, metadata: dict[str, Any]) -> dict[str, Any]:
    checks = [
        predictor_leakage_check(),
        class_balance_check(frame),
        entity_coverage_check(frame),
        temporal_order_check(frame),
        historical_count_consistency(frame),
        split_overlap_check(metadata),
        split_size_check(metadata),
        experiment_space_check(metadata),
        metric_completeness_check(metadata),
    ]
    passed = all(row["passed"] for row in checks)
    return {
        "passed": passed,
        "checks_passed": int(sum(row["passed"] for row in checks)),
        "checks_total": int(len(checks)),
        "checks": checks,
        "protocol": {
            "target": TARGET_COLUMN,
            "predictors": NUMERIC_FEATURES + CATEGORICAL_FEATURES,
            "feature_count": len(NUMERIC_FEATURES) + len(CATEGORICAL_FEATURES),
            "principles": [
                "features are calculated before the current outcome is consumed",
                "learner groups are separated between train and holdout when group split is available",
                "several model families are compared under the same metric set",
                "probability calibration is evaluated alongside ranking quality",
                "temporal backtesting is available as a separate robustness procedure",
            ],
        },
    }
