from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import numpy as np
import pandas as pd

from .feature_engineering import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET_COLUMN


@dataclass(frozen=True)
class QualityCheck:
    name: str
    status: str
    value: Any
    threshold: Any
    message: str


@dataclass(frozen=True)
class QualityReport:
    rows: int
    users: int
    tasks: int
    positive_rate: float
    checks: list[QualityCheck]

    @property
    def passed(self) -> bool:
        return all(check.status != "fail" for check in self.checks)

    def as_dict(self) -> dict[str, Any]:
        return {
            "rows": self.rows,
            "users": self.users,
            "tasks": self.tasks,
            "positive_rate": self.positive_rate,
            "passed": self.passed,
            "checks": [asdict(check) for check in self.checks],
        }


def _check(name: str, condition: bool, value: Any, threshold: Any, ok_message: str, fail_message: str) -> QualityCheck:
    return QualityCheck(
        name=name,
        status="pass" if condition else "fail",
        value=value,
        threshold=threshold,
        message=ok_message if condition else fail_message,
    )


def inspect_training_frame(frame: pd.DataFrame) -> QualityReport:
    checks: list[QualityCheck] = []
    rows = len(frame)
    users = int(frame["user_id"].nunique()) if "user_id" in frame else 0
    tasks = int(frame["task_id"].nunique()) if "task_id" in frame else 0
    positive_rate = float(frame[TARGET_COLUMN].mean()) if rows and TARGET_COLUMN in frame else 0.0

    checks.append(_check("row_count", rows >= 250, rows, ">=250", "Dataset size is sufficient for the configured gate.", "Too few observations for stable model comparison."))
    checks.append(_check("user_count", users >= 5, users, ">=5", "Multiple users are represented.", "Group-aware validation is unreliable with fewer than five users."))
    checks.append(_check("task_count", tasks >= 12, tasks, ">=12", "Task diversity is sufficient.", "Too few tasks are represented in the training history."))
    checks.append(_check("class_balance", 0.10 <= positive_rate <= 0.90, round(positive_rate, 4), "0.10..0.90", "Both outcome classes have useful support.", "Outcome distribution is too imbalanced."))

    missing_features = [name for name in NUMERIC_FEATURES + CATEGORICAL_FEATURES if name not in frame.columns]
    checks.append(_check("feature_schema", not missing_features, missing_features, "none", "All expected features are present.", "Feature schema is incomplete."))

    if not missing_features:
        numeric = frame[NUMERIC_FEATURES].apply(pd.to_numeric, errors="coerce")
        non_finite = int((~np.isfinite(numeric.to_numpy(dtype=float))).sum())
        checks.append(_check("finite_numeric", non_finite == 0, non_finite, 0, "Numeric features are finite.", "NaN or infinite numeric features detected."))

        constant = [column for column in NUMERIC_FEATURES if numeric[column].nunique(dropna=False) <= 1]
        status = "warn" if constant else "pass"
        checks.append(QualityCheck(
            name="constant_features",
            status=status,
            value=constant,
            threshold="prefer none",
            message="Some features are constant in this dataset." if constant else "No constant numeric features detected.",
        ))

    duplicate_attempts = int(frame["attempt_id"].duplicated().sum()) if "attempt_id" in frame else 0
    checks.append(_check("duplicate_attempt_ids", duplicate_attempts == 0, duplicate_attempts, 0, "Attempt identifiers are unique.", "Duplicate attempt identifiers may bias evaluation."))

    chronological = True
    if "created_at" in frame:
        timestamps = pd.to_datetime(frame["created_at"], errors="coerce")
        chronological = bool(timestamps.notna().all())
    checks.append(_check("valid_timestamps", chronological, chronological, True, "Attempt timestamps are parseable.", "Invalid timestamps prevent temporal diagnostics."))

    return QualityReport(rows=rows, users=users, tasks=tasks, positive_rate=positive_rate, checks=checks)


def assert_training_quality(frame: pd.DataFrame) -> QualityReport:
    report = inspect_training_frame(frame)
    failed = [check for check in report.checks if check.status == "fail"]
    if failed:
        details = "; ".join(f"{item.name}: {item.message}" for item in failed)
        raise ValueError(f"training data quality gate failed: {details}")
    return report
