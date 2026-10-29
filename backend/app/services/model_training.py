from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ..config import get_settings
from ..timeutils import utcnow_naive
from .calibration import calibration_report
from .feature_engineering import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET_COLUMN, validate_feature_frame
from .model_experiments import ExperimentSpec, experiment_specs
from .model_selection_report import build_selection_report


@dataclass
class CandidateResult:
    name: str
    family: str
    parameters: dict[str, Any]
    fold_metrics: list[dict[str, float]]
    validation_metrics: dict[str, float]
    validation_calibration: dict[str, Any]
    composite_score: float
    estimator: Pipeline


def _preprocessor(scale_numeric: bool) -> ColumnTransformer:
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    numeric = Pipeline(numeric_steps)
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("numeric", numeric, NUMERIC_FEATURES),
        ("categorical", categorical, CATEGORICAL_FEATURES),
    ])


def pipeline_for_spec(spec: ExperimentSpec) -> Pipeline:
    return Pipeline([
        ("preprocess", _preprocessor(spec.scale_numeric)),
        ("model", spec.estimator),
    ])


def candidate_estimators(random_seed: int, specs: Iterable[ExperimentSpec] | None = None) -> dict[str, Pipeline]:
    selected = list(specs) if specs is not None else experiment_specs(random_seed)
    return {spec.name: pipeline_for_spec(spec) for spec in selected}


def probability_metrics(y_true: np.ndarray, probability: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    probability = np.clip(np.asarray(probability, dtype=float), 1e-6, 1.0 - 1e-6)
    prediction = (probability >= threshold).astype(int)
    result = {
        "accuracy": float(accuracy_score(y_true, prediction)),
        "precision": float(precision_score(y_true, prediction, zero_division=0)),
        "recall": float(recall_score(y_true, prediction, zero_division=0)),
        "f1": float(f1_score(y_true, prediction, zero_division=0)),
        "log_loss": float(log_loss(y_true, np.column_stack([1.0 - probability, probability]), labels=[0, 1])),
        "brier": float(brier_score_loss(y_true, probability)),
    }
    if len(np.unique(y_true)) > 1:
        result["roc_auc"] = float(roc_auc_score(y_true, probability))
        result["pr_auc"] = float(average_precision_score(y_true, probability))
    else:
        result["roc_auc"] = 0.5
        result["pr_auc"] = float(np.mean(y_true))
    return result


def composite_metric(metrics: dict[str, float]) -> float:
    calibration_component = max(0.0, 1.0 - metrics["brier"])
    log_component = 1.0 / (1.0 + metrics["log_loss"])
    return float(
        0.30 * metrics["roc_auc"]
        + 0.20 * metrics["pr_auc"]
        + 0.20 * calibration_component
        + 0.15 * metrics["f1"]
        + 0.15 * log_component
    )


def _dataset_hash(frame: pd.DataFrame) -> str:
    stable = frame.sort_values(["created_at", "attempt_id"]).copy()
    payload = stable.to_json(orient="records", date_format="iso", double_precision=8)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _cross_validate(estimator: Pipeline, x: pd.DataFrame, y: np.ndarray, groups: np.ndarray) -> list[dict[str, float]]:
    unique_groups = np.unique(groups)
    n_splits = min(3, len(unique_groups))
    if n_splits < 2:
        return []
    splitter = GroupKFold(n_splits=n_splits)
    metrics: list[dict[str, float]] = []
    for train_idx, valid_idx in splitter.split(x, y, groups=groups):
        estimator.fit(x.iloc[train_idx], y[train_idx])
        p = estimator.predict_proba(x.iloc[valid_idx])[:, 1]
        metrics.append(probability_metrics(y[valid_idx], p))
    return metrics


def _mean_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = sorted(rows[0])
    return {key: float(np.mean([row[key] for row in rows])) for key in keys}


def _split_indices(frame: pd.DataFrame, groups: np.ndarray, random_seed: int) -> tuple[np.ndarray, np.ndarray, str]:
    unique_users = np.unique(groups)
    if len(unique_users) >= 5:
        split = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=random_seed)
        train_idx, valid_idx = next(split.split(frame, frame[TARGET_COLUMN].to_numpy(), groups=groups))
        return train_idx, valid_idx, "group_holdout"
    cut = max(2, int(len(frame) * 0.75))
    train_idx = np.arange(cut)
    valid_idx = np.arange(cut, len(frame))
    if len(valid_idx) == 0:
        valid_idx = train_idx[-1:]
        train_idx = train_idx[:-1]
    return train_idx, valid_idx, "chronological_fallback"


def train_and_compare(
    frame: pd.DataFrame,
    artifact_dir: Path | None = None,
    *,
    specs: Iterable[ExperimentSpec] | None = None,
) -> dict[str, Any]:
    settings = get_settings()
    validate_feature_frame(frame, require_target=True)
    if frame[TARGET_COLUMN].nunique() < 2:
        raise ValueError("training data must contain both successful and unsuccessful attempts")
    if frame["user_id"].nunique() < 3:
        raise ValueError("training data must contain at least three users")

    selected_specs = list(specs) if specs is not None else experiment_specs(settings.random_seed)
    if not selected_specs:
        raise ValueError("at least one experiment specification is required")
    spec_by_name = {spec.name: spec for spec in selected_specs}

    features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    x = frame[features].copy()
    y = frame[TARGET_COLUMN].astype(int).to_numpy()
    groups = frame["user_id"].to_numpy()
    train_idx, valid_idx, split_strategy = _split_indices(frame, groups, settings.random_seed)
    train_users = sorted(int(value) for value in np.unique(groups[train_idx]))
    valid_users = sorted(int(value) for value in np.unique(groups[valid_idx]))
    group_overlap = sorted(set(train_users).intersection(valid_users))

    candidates: list[CandidateResult] = []
    for name, estimator in candidate_estimators(settings.random_seed, selected_specs).items():
        spec = spec_by_name[name]
        cv_rows = _cross_validate(estimator, x.iloc[train_idx], y[train_idx], groups[train_idx])
        estimator.fit(x.iloc[train_idx], y[train_idx])
        valid_p = estimator.predict_proba(x.iloc[valid_idx])[:, 1]
        validation = probability_metrics(y[valid_idx], valid_p)
        calibration = calibration_report(y[valid_idx], valid_p, bins=8)
        cv_mean = _mean_metrics(cv_rows)
        blended = dict(validation)
        if cv_mean:
            for key, value in validation.items():
                blended[key] = 0.65 * value + 0.35 * cv_mean.get(key, value)
        score = composite_metric(blended)
        candidates.append(CandidateResult(
            name=name,
            family=spec.family,
            parameters=dict(spec.parameters),
            fold_metrics=cv_rows,
            validation_metrics=validation,
            validation_calibration=calibration,
            composite_score=score,
            estimator=estimator,
        ))

    candidates.sort(key=lambda item: item.composite_score, reverse=True)
    best = candidates[0]
    best.estimator.fit(x, y)

    run_id = uuid.uuid4().hex[:16]
    version = f"{best.name}-{utcnow_naive().strftime('%Y%m%d%H%M%S')}-{run_id[:6]}"
    artifact_dir = artifact_dir or settings.model_dir
    artifact_dir.mkdir(parents=True, exist_ok=True)
    model_path = artifact_dir / f"{version}.joblib"
    metadata_path = artifact_dir / f"{version}.json"
    joblib.dump(best.estimator, model_path)

    candidate_rows = [
        {
            "name": item.name,
            "family": item.family,
            "parameters": item.parameters,
            "composite_score": item.composite_score,
            "validation": item.validation_metrics,
            "validation_calibration": {
                "ece": item.validation_calibration["ece"],
                "mce": item.validation_calibration["mce"],
                "slope": item.validation_calibration["slope"],
                "intercept": item.validation_calibration["intercept"],
            },
            "cross_validation_mean": _mean_metrics(item.fold_metrics),
            "cross_validation": item.fold_metrics,
        }
        for item in candidates
    ]
    metadata = {
        "run_id": run_id,
        "version": version,
        "best_model": best.name,
        "best_family": best.family,
        "best_parameters": best.parameters,
        "trained_at": utcnow_naive().isoformat() + "Z",
        "dataset_rows": int(len(frame)),
        "users": int(frame["user_id"].nunique()),
        "positive_rate": float(frame[TARGET_COLUMN].mean()),
        "dataset_hash": _dataset_hash(frame),
        "feature_names": features,
        "feature_count": len(features),
        "selection_metric": "weighted ROC-AUC / PR-AUC / Brier / F1 / log-loss",
        "experiment_count": len(candidates),
        "experiment_space": [
            {"name": spec.name, "family": spec.family, "parameters": spec.parameters}
            for spec in selected_specs
        ],
        "split": {
            "strategy": split_strategy,
            "train_rows": int(len(train_idx)),
            "validation_rows": int(len(valid_idx)),
            "train_users": train_users,
            "validation_users": valid_users,
            "user_overlap": group_overlap,
        },
        "candidates": candidate_rows,
        "selection_report": build_selection_report(candidate_rows),
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "run_id": run_id,
        "version": version,
        "model_name": best.name,
        "model_path": str(model_path),
        "metadata_path": str(metadata_path),
        "metadata": metadata,
        "estimator": best.estimator,
    }
