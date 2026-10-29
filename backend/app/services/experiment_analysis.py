from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GroupKFold

from ..config import get_settings
from .feature_engineering import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET_COLUMN, validate_feature_frame
from .model_training import probability_metrics, composite_metric
from .statistics_utils import bootstrap_confidence_interval, bootstrap_difference_interval


@dataclass(frozen=True)
class FoldResult:
    model: str
    fold: int
    train_rows: int
    validation_rows: int
    validation_users: int
    metrics: dict[str, float]

    def as_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "fold": self.fold,
            "train_rows": self.train_rows,
            "validation_rows": self.validation_rows,
            "validation_users": self.validation_users,
            "metrics": self.metrics,
            "composite_score": composite_metric(self.metrics),
        }


def repeated_group_benchmark(
    frame: pd.DataFrame,
    estimators: dict[str, Any],
    *,
    folds: int = 3,
    repetitions: int = 1,
    random_seed: int | None = None,
) -> dict[str, Any]:
    validate_feature_frame(frame, require_target=True)
    if frame["user_id"].nunique() < 3:
        raise ValueError("benchmark requires at least three users")
    features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    x = frame[features]
    y = frame[TARGET_COLUMN].astype(int).to_numpy()
    groups = frame["user_id"].to_numpy()
    unique_users = np.unique(groups)
    folds = min(max(2, int(folds)), len(unique_users))
    repetitions = max(1, int(repetitions))
    base_seed = get_settings().random_seed if random_seed is None else random_seed

    results: list[FoldResult] = []
    prediction_store: dict[str, list[tuple[int, float, int]]] = defaultdict(list)
    for repetition in range(repetitions):
        user_order = unique_users.copy()
        rng = np.random.default_rng(base_seed + repetition)
        rng.shuffle(user_order)
        remap = {user: index for index, user in enumerate(user_order)}
        ordered_groups = np.asarray([remap[group] for group in groups])
        splitter = GroupKFold(n_splits=folds)
        for fold_index, (train_idx, validation_idx) in enumerate(splitter.split(x, y, groups=ordered_groups), start=1):
            for model_name, estimator in estimators.items():
                fitted = clone(estimator)
                fitted.fit(x.iloc[train_idx], y[train_idx])
                probability = fitted.predict_proba(x.iloc[validation_idx])[:, 1]
                metrics = probability_metrics(y[validation_idx], probability)
                results.append(FoldResult(
                    model=model_name,
                    fold=repetition * folds + fold_index,
                    train_rows=len(train_idx),
                    validation_rows=len(validation_idx),
                    validation_users=len(np.unique(groups[validation_idx])),
                    metrics=metrics,
                ))
                for local_idx, probability_value in zip(validation_idx, probability):
                    prediction_store[model_name].append((int(local_idx), float(probability_value), int(y[local_idx])))

    summaries = _summarize_folds(results)
    comparisons = _pairwise_comparisons(prediction_store, iterations=600, seed=base_seed)
    return {
        "rows": int(len(frame)),
        "users": int(frame["user_id"].nunique()),
        "folds": folds,
        "repetitions": repetitions,
        "models": list(estimators),
        "fold_results": [item.as_dict() for item in results],
        "summary": summaries,
        "pairwise_brier_comparisons": comparisons,
    }


def _summarize_folds(results: Iterable[FoldResult]) -> list[dict[str, Any]]:
    grouped: dict[str, list[FoldResult]] = defaultdict(list)
    for row in results:
        grouped[row.model].append(row)
    summary: list[dict[str, Any]] = []
    for model_name, rows in grouped.items():
        metric_names = sorted(rows[0].metrics)
        metrics: dict[str, Any] = {}
        for metric in metric_names:
            values = [row.metrics[metric] for row in rows]
            interval = bootstrap_confidence_interval(values, iterations=800)
            metrics[metric] = interval.as_dict()
        composite_values = [composite_metric(row.metrics) for row in rows]
        composite_ci = bootstrap_confidence_interval(composite_values, iterations=800)
        summary.append({
            "model": model_name,
            "fold_count": len(rows),
            "composite": composite_ci.as_dict(),
            "metrics": metrics,
        })
    return sorted(summary, key=lambda item: item["composite"]["estimate"], reverse=True)


def _pairwise_comparisons(
    prediction_store: dict[str, list[tuple[int, float, int]]],
    *,
    iterations: int,
    seed: int,
) -> list[dict[str, Any]]:
    model_names = sorted(prediction_store)
    output: list[dict[str, Any]] = []
    indexed = {
        name: {row_id: (probability, target) for row_id, probability, target in rows}
        for name, rows in prediction_store.items()
    }
    for left_index, left_name in enumerate(model_names):
        for right_name in model_names[left_index + 1:]:
            common = sorted(set(indexed[left_name]).intersection(indexed[right_name]))
            if not common:
                continue
            left_loss = []
            right_loss = []
            for row_id in common:
                left_p, y = indexed[left_name][row_id]
                right_p, right_y = indexed[right_name][row_id]
                if y != right_y:
                    continue
                left_loss.append((left_p - y) ** 2)
                right_loss.append((right_p - y) ** 2)
            interval = bootstrap_difference_interval(
                left_loss,
                right_loss,
                confidence=0.95,
                iterations=iterations,
                random_seed=seed + left_index,
            )
            if interval.upper < 0:
                interpretation = f"{left_name} has lower Brier loss"
            elif interval.lower > 0:
                interpretation = f"{right_name} has lower Brier loss"
            else:
                interpretation = "difference is inconclusive at the 95% bootstrap interval"
            output.append({
                "left": left_name,
                "right": right_name,
                "paired_rows": len(left_loss),
                "mean_brier_loss_difference_left_minus_right": interval.as_dict(),
                "interpretation": interpretation,
            })
    return output
