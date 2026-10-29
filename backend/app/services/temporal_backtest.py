from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone

from .calibration import calibration_report
from .feature_engineering import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET_COLUMN, validate_feature_frame
from .model_training import probability_metrics
from .statistics_utils import bootstrap_confidence_interval


@dataclass(frozen=True)
class BacktestWindow:
    window: int
    train_rows: int
    test_rows: int
    train_users: int
    test_users: int
    train_end: str
    test_start: str
    test_end: str
    metrics: dict[str, float]
    ece: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "window": self.window,
            "train_rows": self.train_rows,
            "test_rows": self.test_rows,
            "train_users": self.train_users,
            "test_users": self.test_users,
            "train_end": self.train_end,
            "test_start": self.test_start,
            "test_end": self.test_end,
            "metrics": self.metrics,
            "ece": self.ece,
        }


def expanding_window_backtest(
    estimator,
    frame: pd.DataFrame,
    *,
    windows: int = 4,
    minimum_train_rows: int = 180,
    minimum_test_rows: int = 40,
) -> dict[str, Any]:
    validate_feature_frame(frame, require_target=True)
    if "created_at" not in frame.columns:
        raise ValueError("created_at is required for temporal backtesting")
    ordered = frame.copy()
    ordered["created_at"] = pd.to_datetime(ordered["created_at"], errors="raise")
    ordered = ordered.sort_values(["created_at", "attempt_id"]).reset_index(drop=True)
    n = len(ordered)
    if n < minimum_train_rows + minimum_test_rows:
        raise ValueError("not enough rows for temporal backtesting")
    windows = max(2, min(int(windows), 8))
    features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    boundaries = _window_boundaries(n, windows, minimum_train_rows, minimum_test_rows)
    results: list[BacktestWindow] = []
    for index, (train_end_index, test_end_index) in enumerate(boundaries, start=1):
        train = ordered.iloc[:train_end_index]
        test = ordered.iloc[train_end_index:test_end_index]
        if len(test) < minimum_test_rows or train[TARGET_COLUMN].nunique() < 2 or test[TARGET_COLUMN].nunique() < 2:
            continue
        model = clone(estimator)
        model.fit(train[features], train[TARGET_COLUMN].astype(int).to_numpy())
        probability = model.predict_proba(test[features])[:, 1]
        y = test[TARGET_COLUMN].astype(int).to_numpy()
        metrics = probability_metrics(y, probability)
        calibration = calibration_report(y, probability, bins=8)
        results.append(BacktestWindow(
            window=index,
            train_rows=len(train),
            test_rows=len(test),
            train_users=int(train["user_id"].nunique()),
            test_users=int(test["user_id"].nunique()),
            train_end=train["created_at"].iloc[-1].isoformat(),
            test_start=test["created_at"].iloc[0].isoformat(),
            test_end=test["created_at"].iloc[-1].isoformat(),
            metrics=metrics,
            ece=float(calibration["ece"]),
        ))
    if not results:
        raise ValueError("no valid temporal backtest windows could be constructed")
    return {
        "strategy": "expanding-window chronological validation",
        "rows": n,
        "window_count": len(results),
        "windows": [row.as_dict() for row in results],
        "summary": _summary(results),
        "leakage_note": "Every validation observation occurs after the observations used to fit that window.",
    }


def _window_boundaries(n: int, windows: int, minimum_train: int, minimum_test: int) -> list[tuple[int, int]]:
    remaining = n - minimum_train
    nominal = max(minimum_test, remaining // windows)
    output: list[tuple[int, int]] = []
    train_end = minimum_train
    while train_end + minimum_test <= n and len(output) < windows:
        test_end = min(n, train_end + nominal)
        if test_end - train_end < minimum_test:
            break
        output.append((train_end, test_end))
        train_end = test_end
    if output and output[-1][1] < n and n - output[-1][0] >= minimum_test:
        output[-1] = (output[-1][0], n)
    return output


def _summary(results: list[BacktestWindow]) -> dict[str, Any]:
    metric_names = sorted(results[0].metrics)
    summary: dict[str, Any] = {}
    for name in metric_names:
        values = [row.metrics[name] for row in results]
        summary[name] = bootstrap_confidence_interval(values, iterations=800).as_dict()
    summary["ece"] = bootstrap_confidence_interval([row.ece for row in results], iterations=800).as_dict()
    if len(results) > 1:
        first = results[0].metrics["brier"]
        last = results[-1].metrics["brier"]
        summary["brier_change_first_to_last"] = float(last - first)
        summary["roc_auc_change_first_to_last"] = float(results[-1].metrics["roc_auc"] - results[0].metrics["roc_auc"])
    return summary
