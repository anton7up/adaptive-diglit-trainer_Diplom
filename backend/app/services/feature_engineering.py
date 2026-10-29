from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Iterable

import numpy as np
import pandas as pd

from .. import models
from ..timeutils import utcnow_naive

NUMERIC_FEATURES = [
    "user_attempts_total",
    "user_accuracy_total",
    "user_accuracy_last5",
    "user_accuracy_last10",
    "user_attempts_24h",
    "user_attempts_7d",
    "skill_attempts",
    "skill_accuracy",
    "skill_accuracy_last5",
    "skill_accuracy_last10",
    "skill_attempts_7d",
    "task_attempts",
    "task_accuracy",
    "task_exposure_share",
    "skill_exposure_share",
    "seconds_since_last_attempt",
    "seconds_since_last_skill_attempt",
    "seconds_since_last_task_attempt",
    "response_time_mean_sec",
    "response_time_std_sec",
    "response_time_last_sec",
    "skill_response_time_mean_sec",
    "skill_response_time_last_sec",
    "correct_streak",
    "incorrect_streak",
    "skill_correct_streak",
    "skill_incorrect_streak",
    "task_difficulty",
    "task_discrimination",
    "task_cognitive_level",
    "task_estimated_time_sec",
    "difficulty_gap",
    "speed_ratio",
    "practice_density_7d",
    "local_hour_sin",
    "local_hour_cos",
    "is_first_skill_attempt",
    "is_first_task_attempt",
]
CATEGORICAL_FEATURES = ["skill_code"]
TARGET_COLUMN = "target"


@dataclass
class FeatureState:
    attempts_total: int = 0
    correct_total: int = 0
    response_sum_ms: int = 0
    response_sq_sum_ms: float = 0.0
    last_attempt_at: datetime | None = None
    last_response_ms: int = 0
    correct_streak: int = 0
    incorrect_streak: int = 0


@dataclass
class SkillState:
    attempts: int = 0
    correct: int = 0
    response_sum_ms: int = 0
    last_response_ms: int = 0
    last_attempt_at: datetime | None = None
    correct_streak: int = 0
    incorrect_streak: int = 0


class HistoryAccumulator:
    """Builds pre-attempt features while preserving temporal causality."""

    def __init__(self) -> None:
        self.user_state: dict[int, FeatureState] = defaultdict(FeatureState)
        self.skill_state: dict[tuple[int, int], SkillState] = defaultdict(SkillState)
        self.user_recent: dict[int, deque[int]] = defaultdict(lambda: deque(maxlen=10))
        self.skill_recent: dict[tuple[int, int], deque[int]] = defaultdict(lambda: deque(maxlen=10))
        self.user_timestamps: dict[int, deque[datetime]] = defaultdict(deque)
        self.skill_timestamps: dict[tuple[int, int], deque[datetime]] = defaultdict(deque)
        self.task_attempts: dict[tuple[int, int], int] = defaultdict(int)
        self.task_correct: dict[tuple[int, int], int] = defaultdict(int)
        self.task_last_at: dict[tuple[int, int], datetime] = {}

    @staticmethod
    def _ratio(numerator: int, denominator: int, prior: float = 0.5, strength: float = 2.0) -> float:
        return float((numerator + prior * strength) / (denominator + strength))

    @staticmethod
    def _window_accuracy(values: deque[int], length: int) -> float:
        window = list(values)[-length:]
        return float(sum(window) / len(window)) if window else 0.5

    @staticmethod
    def _seconds_since(previous: datetime | None, current: datetime, cap_days: float = 60.0) -> float:
        if previous is None:
            return cap_days * 86400.0
        return float(min(max(0.0, (current - previous).total_seconds()), cap_days * 86400.0))

    @staticmethod
    def _prune_window(values: deque[datetime], at: datetime, window: timedelta) -> None:
        cutoff = at - window
        while values and values[0] < cutoff:
            values.popleft()

    @staticmethod
    def _time_cyclical(at: datetime) -> tuple[float, float]:
        radians = 2.0 * np.pi * ((at.hour * 60 + at.minute) / (24.0 * 60.0))
        return float(np.sin(radians)), float(np.cos(radians))

    def features(
        self,
        user_id: int,
        task: models.Task,
        skill_code: str,
        at: datetime,
    ) -> dict[str, float | int | str]:
        state = self.user_state[user_id]
        skey = (user_id, task.skill_id)
        tkey = (user_id, task.id)
        skill = self.skill_state[skey]

        user_recent = self.user_recent[user_id]
        skill_recent = self.skill_recent[skey]
        user_times = self.user_timestamps[user_id]
        skill_times = self.skill_timestamps[skey]
        self._prune_window(user_times, at, timedelta(days=7))
        self._prune_window(skill_times, at, timedelta(days=7))

        user_24h = sum(1 for value in user_times if value >= at - timedelta(days=1))
        user_7d = len(user_times)
        skill_7d = len(skill_times)

        overall_accuracy = self._ratio(state.correct_total, state.attempts_total)
        skill_accuracy = self._ratio(skill.correct, skill.attempts)
        task_n = self.task_attempts[tkey]
        task_accuracy = self._ratio(self.task_correct[tkey], task_n)
        user_last5 = self._window_accuracy(user_recent, 5)
        user_last10 = self._window_accuracy(user_recent, 10)
        skill_last5 = self._window_accuracy(skill_recent, 5)
        skill_last10 = self._window_accuracy(skill_recent, 10)

        mean_response_sec = state.response_sum_ms / state.attempts_total / 1000.0 if state.attempts_total else 0.0
        if state.attempts_total > 1:
            mean_ms = state.response_sum_ms / state.attempts_total
            variance_ms = max(0.0, state.response_sq_sum_ms / state.attempts_total - mean_ms**2)
            response_std_sec = float(np.sqrt(variance_ms) / 1000.0)
        else:
            response_std_sec = 0.0

        skill_response_mean_sec = skill.response_sum_ms / skill.attempts / 1000.0 if skill.attempts else 0.0
        expected_time = max(float(task.estimated_time_sec), 1.0)
        speed_ratio = (state.last_response_ms / 1000.0) / expected_time if state.last_response_ms else 1.0

        mastery_estimate = 0.50 * skill_accuracy + 0.30 * skill_last10 + 0.20 * user_last10
        total_exposures = max(state.attempts_total, 1)
        task_exposure_share = task_n / total_exposures
        skill_exposure_share = skill.attempts / total_exposures
        practice_density = user_7d / 7.0
        hour_sin, hour_cos = self._time_cyclical(at)

        return {
            "user_attempts_total": state.attempts_total,
            "user_accuracy_total": overall_accuracy,
            "user_accuracy_last5": user_last5,
            "user_accuracy_last10": user_last10,
            "user_attempts_24h": user_24h,
            "user_attempts_7d": user_7d,
            "skill_attempts": skill.attempts,
            "skill_accuracy": skill_accuracy,
            "skill_accuracy_last5": skill_last5,
            "skill_accuracy_last10": skill_last10,
            "skill_attempts_7d": skill_7d,
            "task_attempts": task_n,
            "task_accuracy": task_accuracy,
            "task_exposure_share": float(task_exposure_share),
            "skill_exposure_share": float(skill_exposure_share),
            "seconds_since_last_attempt": self._seconds_since(state.last_attempt_at, at),
            "seconds_since_last_skill_attempt": self._seconds_since(skill.last_attempt_at, at),
            "seconds_since_last_task_attempt": self._seconds_since(self.task_last_at.get(tkey), at),
            "response_time_mean_sec": float(mean_response_sec),
            "response_time_std_sec": response_std_sec,
            "response_time_last_sec": state.last_response_ms / 1000.0,
            "skill_response_time_mean_sec": float(skill_response_mean_sec),
            "skill_response_time_last_sec": skill.last_response_ms / 1000.0,
            "correct_streak": state.correct_streak,
            "incorrect_streak": state.incorrect_streak,
            "skill_correct_streak": skill.correct_streak,
            "skill_incorrect_streak": skill.incorrect_streak,
            "task_difficulty": float(task.difficulty),
            "task_discrimination": float(task.discrimination),
            "task_cognitive_level": int(task.cognitive_level),
            "task_estimated_time_sec": int(task.estimated_time_sec),
            "difficulty_gap": float(task.difficulty) - mastery_estimate,
            "speed_ratio": float(min(max(speed_ratio, 0.0), 10.0)),
            "practice_density_7d": float(practice_density),
            "local_hour_sin": hour_sin,
            "local_hour_cos": hour_cos,
            "is_first_skill_attempt": int(skill.attempts == 0),
            "is_first_task_attempt": int(task_n == 0),
            "skill_code": skill_code,
        }

    def consume(self, attempt: models.Attempt) -> None:
        task = attempt.task
        state = self.user_state[attempt.user_id]
        skey = (attempt.user_id, task.skill_id)
        tkey = (attempt.user_id, task.id)
        skill = self.skill_state[skey]
        y = 1 if attempt.is_correct else 0
        response_ms = max(0, attempt.response_time_ms or 0)

        state.attempts_total += 1
        state.correct_total += y
        state.response_sum_ms += response_ms
        state.response_sq_sum_ms += float(response_ms) ** 2
        state.last_attempt_at = attempt.created_at
        state.last_response_ms = response_ms
        if y:
            state.correct_streak += 1
            state.incorrect_streak = 0
        else:
            state.incorrect_streak += 1
            state.correct_streak = 0

        skill.attempts += 1
        skill.correct += y
        skill.response_sum_ms += response_ms
        skill.last_response_ms = response_ms
        skill.last_attempt_at = attempt.created_at
        if y:
            skill.correct_streak += 1
            skill.incorrect_streak = 0
        else:
            skill.incorrect_streak += 1
            skill.correct_streak = 0

        self.user_recent[attempt.user_id].append(y)
        self.skill_recent[skey].append(y)
        self.user_timestamps[attempt.user_id].append(attempt.created_at)
        self.skill_timestamps[skey].append(attempt.created_at)
        self.task_attempts[tkey] += 1
        self.task_correct[tkey] += y
        self.task_last_at[tkey] = attempt.created_at


def build_training_frame(attempts: Iterable[models.Attempt]) -> pd.DataFrame:
    ordered = sorted(attempts, key=lambda row: (row.created_at, row.id or 0))
    accumulator = HistoryAccumulator()
    rows: list[dict[str, object]] = []
    for attempt in ordered:
        if attempt.task is None or attempt.task.skill is None:
            continue
        row = accumulator.features(
            user_id=attempt.user_id,
            task=attempt.task,
            skill_code=attempt.task.skill.code,
            at=attempt.created_at,
        )
        row[TARGET_COLUMN] = int(attempt.is_correct)
        row["user_id"] = attempt.user_id
        row["task_id"] = attempt.task_id
        row["attempt_id"] = attempt.id
        row["created_at"] = attempt.created_at
        rows.append(row)
        accumulator.consume(attempt)
    return pd.DataFrame(rows)


def build_candidate_frame(db, user_id: int, tasks: list[models.Task], now: datetime | None = None) -> pd.DataFrame:
    now = now or utcnow_naive()
    attempts = (
        db.query(models.Attempt)
        .filter(models.Attempt.user_id == user_id)
        .order_by(models.Attempt.created_at.asc(), models.Attempt.id.asc())
        .all()
    )
    accumulator = HistoryAccumulator()
    for attempt in attempts:
        accumulator.consume(attempt)
    return pd.DataFrame([
        accumulator.features(user_id, task, task.skill.code, now)
        for task in tasks
    ])


def validate_feature_frame(frame: pd.DataFrame, require_target: bool = False) -> None:
    expected = set(NUMERIC_FEATURES + CATEGORICAL_FEATURES)
    if require_target:
        expected.add(TARGET_COLUMN)
    missing = expected.difference(frame.columns)
    if missing:
        raise ValueError(f"feature frame misses columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("feature frame is empty")
    for column in NUMERIC_FEATURES:
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.isna().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
            raise ValueError(f"feature {column} contains non-finite values")
