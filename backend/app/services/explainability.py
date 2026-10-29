from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .feature_engineering import CATEGORICAL_FEATURES, NUMERIC_FEATURES


FEATURE_LABELS = {
    "user_accuracy_total": "общая точность",
    "user_accuracy_last5": "точность последних 5 ответов",
    "user_accuracy_last10": "точность последних 10 ответов",
    "skill_accuracy": "точность по компетенции",
    "skill_accuracy_last5": "последние ответы по компетенции",
    "skill_accuracy_last10": "устойчивость результата по компетенции",
    "task_accuracy": "историческая успешность на этом задании",
    "task_difficulty": "сложность задания",
    "task_discrimination": "различающая способность задания",
    "task_cognitive_level": "когнитивный уровень задания",
    "difficulty_gap": "разрыв между текущим уровнем и сложностью",
    "seconds_since_last_skill_attempt": "время с последней практики компетенции",
    "response_time_mean_sec": "среднее время ответа",
    "response_time_last_sec": "время последнего ответа",
    "correct_streak": "серия правильных ответов",
    "incorrect_streak": "серия ошибок",
    "skill_correct_streak": "серия правильных ответов по компетенции",
    "skill_incorrect_streak": "серия ошибок по компетенции",
    "practice_density_7d": "плотность практики за неделю",
    "skill_exposure_share": "доля практики этой компетенции",
    "task_exposure_share": "частота повторения задания",
}


@dataclass(frozen=True)
class LocalContribution:
    feature: str
    label: str
    value: Any
    reference: Any
    probability_delta: float
    direction: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "feature": self.feature,
            "label": self.label,
            "value": self.value,
            "reference": self.reference,
            "probability_delta": self.probability_delta,
            "direction": self.direction,
        }


def _prediction(estimator, frame: pd.DataFrame) -> np.ndarray:
    return np.asarray(estimator.predict_proba(frame[NUMERIC_FEATURES + CATEGORICAL_FEATURES])[:, 1], dtype=float)


def reference_profile(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        raise ValueError("reference frame is empty")
    result: dict[str, Any] = {}
    for feature in NUMERIC_FEATURES:
        values = pd.to_numeric(frame[feature], errors="coerce")
        finite = values[np.isfinite(values)]
        result[feature] = float(finite.median()) if len(finite) else 0.0
    for feature in CATEGORICAL_FEATURES:
        mode = frame[feature].dropna().mode()
        result[feature] = mode.iloc[0] if len(mode) else "UNKNOWN"
    return result


def explain_row(
    estimator,
    row: pd.DataFrame,
    reference: dict[str, Any],
    *,
    max_features: int = 8,
) -> dict[str, Any]:
    if len(row) != 1:
        raise ValueError("local explanation expects exactly one row")
    base_probability = float(_prediction(estimator, row)[0])
    contributions: list[LocalContribution] = []
    for feature in NUMERIC_FEATURES + CATEGORICAL_FEATURES:
        modified = row.copy()
        original_value = modified.iloc[0][feature]
        reference_value = reference[feature]
        modified.loc[modified.index[0], feature] = reference_value
        counterfactual_probability = float(_prediction(estimator, modified)[0])
        delta = base_probability - counterfactual_probability
        if abs(delta) < 1e-8:
            continue
        contributions.append(LocalContribution(
            feature=feature,
            label=FEATURE_LABELS.get(feature, feature),
            value=_json_scalar(original_value),
            reference=_json_scalar(reference_value),
            probability_delta=float(delta),
            direction="raises_probability" if delta > 0 else "lowers_probability",
        ))
    contributions.sort(key=lambda item: abs(item.probability_delta), reverse=True)
    selected = contributions[:max(1, max_features)]
    return {
        "predicted_probability": base_probability,
        "top_factors": [item.as_dict() for item in selected],
        "summary": _natural_language_summary(selected),
        "method": "one-feature counterfactual perturbation around the training median/mode",
    }


def explain_batch(
    estimator,
    frame: pd.DataFrame,
    *,
    reference_frame: pd.DataFrame | None = None,
    max_rows: int = 100,
    max_features: int = 6,
) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    reference = reference_profile(reference_frame if reference_frame is not None else frame)
    sample = frame.head(max_rows)
    return [
        {
            "row_index": _json_scalar(index),
            **explain_row(estimator, sample.loc[[index]], reference, max_features=max_features),
        }
        for index in sample.index
    ]


def global_sensitivity(
    estimator,
    frame: pd.DataFrame,
    *,
    max_rows: int = 800,
) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    sample = frame.tail(max_rows).copy()
    baseline = _prediction(estimator, sample)
    reference = reference_profile(sample)
    rows: list[dict[str, Any]] = []
    for feature in NUMERIC_FEATURES + CATEGORICAL_FEATURES:
        modified = sample.copy()
        modified[feature] = reference[feature]
        changed = _prediction(estimator, modified)
        deltas = baseline - changed
        rows.append({
            "feature": feature,
            "label": FEATURE_LABELS.get(feature, feature),
            "mean_absolute_probability_change": float(np.mean(np.abs(deltas))),
            "mean_signed_probability_change": float(np.mean(deltas)),
            "p90_absolute_probability_change": float(np.quantile(np.abs(deltas), 0.90)),
        })
    return sorted(rows, key=lambda item: item["mean_absolute_probability_change"], reverse=True)


def _natural_language_summary(contributions: list[LocalContribution]) -> list[str]:
    messages: list[str] = []
    for item in contributions[:4]:
        direction = "повышает" if item.probability_delta > 0 else "снижает"
        messages.append(
            f"{item.label}: {direction} прогноз примерно на {abs(item.probability_delta) * 100:.1f} п.п. относительно типичного профиля"
        )
    return messages


def _json_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if pd.isna(value):
        return None
    return value
