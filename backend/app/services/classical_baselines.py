from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from math import exp, log
from typing import Any, Callable

import numpy as np
import pandas as pd

from ..config import get_settings
from .feature_engineering import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET_COLUMN, validate_feature_frame
from .model_experiments import experiment_specs
from .model_training import candidate_estimators, probability_metrics
from .statistics_utils import (
    bootstrap_difference_interval,
    holm_adjusted_pvalues,
    paired_randomization_pvalue,
    paired_standardized_effect,
)


def _sigmoid(value: float) -> float:
    value = max(min(float(value), 25.0), -25.0)
    return float(1.0 / (1.0 + exp(-value)))


def _row_key(row: pd.Series) -> tuple[int, str]:
    return int(row["user_id"]), str(row["skill_code"])


class OnlineIRT2PL:
    def __init__(self, learning_rate: float = 0.035) -> None:
        self.learning_rate = float(learning_rate)
        self.theta: dict[tuple[int, str], float] = defaultdict(lambda: 0.50)

    def step(self, row: pd.Series, *, update: bool = True) -> float:
        key = _row_key(row)
        ability = self.theta[key]
        difficulty = float(row["task_difficulty"])
        discrimination = max(float(row["task_discrimination"]), 0.10)
        slope = 5.0 * discrimination
        probability = _sigmoid(slope * (ability - difficulty))
        if update:
            target = int(row[TARGET_COLUMN])
            ability += self.learning_rate * slope * (target - probability)
            self.theta[key] = float(np.clip(ability, 0.02, 0.98))
        return probability


class OnlineBKT:
    def __init__(
        self,
        p_init: float = 0.25,
        p_learn: float = 0.12,
        p_guess: float = 0.20,
        p_slip: float = 0.10,
    ) -> None:
        self.p_init = float(p_init)
        self.p_learn = float(p_learn)
        self.p_guess = float(p_guess)
        self.p_slip = float(p_slip)
        self.mastery: dict[tuple[int, str], float] = defaultdict(lambda: self.p_init)

    def step(self, row: pd.Series, *, update: bool = True) -> float:
        key = _row_key(row)
        mastery = self.mastery[key]
        probability = mastery * (1.0 - self.p_slip) + (1.0 - mastery) * self.p_guess
        if update:
            target = int(row[TARGET_COLUMN])
            if target:
                numerator = mastery * (1.0 - self.p_slip)
                denominator = max(probability, 1e-9)
            else:
                numerator = mastery * self.p_slip
                denominator = max(1.0 - probability, 1e-9)
            posterior = float(np.clip(numerator / denominator, 0.0, 1.0))
            transitioned = posterior + (1.0 - posterior) * self.p_learn
            self.mastery[key] = float(np.clip(transitioned, 0.001, 0.999))
        return float(np.clip(probability, 1e-6, 1.0 - 1e-6))


class OnlinePFA:
    def __init__(
        self,
        intercept: float = -0.35,
        success_weight: float = 0.42,
        failure_weight: float = -0.48,
        difficulty_weight: float = 2.0,
    ) -> None:
        self.intercept = float(intercept)
        self.success_weight = float(success_weight)
        self.failure_weight = float(failure_weight)
        self.difficulty_weight = float(difficulty_weight)
        self.successes: dict[tuple[int, str], int] = defaultdict(int)
        self.failures: dict[tuple[int, str], int] = defaultdict(int)

    def step(self, row: pd.Series, *, update: bool = True) -> float:
        key = _row_key(row)
        score = (
            self.intercept
            + self.success_weight * self.successes[key]
            + self.failure_weight * self.failures[key]
            + self.difficulty_weight * (0.50 - float(row["task_difficulty"]))
        )
        probability = _sigmoid(score)
        if update:
            if int(row[TARGET_COLUMN]):
                self.successes[key] += 1
            else:
                self.failures[key] += 1
        return probability


class OnlineElo:
    def __init__(self, k_user: float = 0.20, k_item: float = 0.08) -> None:
        self.k_user = float(k_user)
        self.k_item = float(k_item)
        self.ability: dict[tuple[int, str], float] = defaultdict(float)
        self.item_rating: dict[int, float] = {}

    def _initial_item_rating(self, row: pd.Series) -> float:
        difficulty = float(np.clip(row["task_difficulty"], 1e-4, 1.0 - 1e-4))
        return float(log(difficulty / (1.0 - difficulty)))

    def step(self, row: pd.Series, *, update: bool = True, update_item: bool = True) -> float:
        key = _row_key(row)
        task_id = int(row["task_id"])
        if task_id not in self.item_rating:
            self.item_rating[task_id] = self._initial_item_rating(row)
        user_rating = self.ability[key]
        item_rating = self.item_rating[task_id]
        probability = _sigmoid(user_rating - item_rating)
        if update:
            error = int(row[TARGET_COLUMN]) - probability
            self.ability[key] = float(np.clip(user_rating + self.k_user * error, -4.0, 4.0))
            if update_item:
                self.item_rating[task_id] = float(np.clip(item_rating - self.k_item * error, -4.0, 4.0))
        return probability


@dataclass(frozen=True)
class BaselineSpec:
    name: str
    family: str
    factory: Callable[[], Any]
    parameters: dict[str, float]


def _ordered(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [column for column in ("created_at", "attempt_id") if column in frame.columns]
    return frame.sort_values(columns, kind="stable") if columns else frame.copy()


def _prequential_predictions(frame: pd.DataFrame, predictor: Any, *, freeze_global_items: bool = False) -> np.ndarray:
    probabilities: list[float] = []
    for _, row in _ordered(frame).iterrows():
        if isinstance(predictor, OnlineElo):
            probability = predictor.step(row, update=True, update_item=not freeze_global_items)
        else:
            probability = predictor.step(row, update=True)
        probabilities.append(float(np.clip(probability, 1e-6, 1.0 - 1e-6)))
    return np.asarray(probabilities, dtype=float)


def _tune(
    train: pd.DataFrame,
    name: str,
    candidates: list[BaselineSpec],
) -> BaselineSpec:
    target = _ordered(train)[TARGET_COLUMN].astype(int).to_numpy()
    best: tuple[float, BaselineSpec] | None = None
    for spec in candidates:
        predictor = spec.factory()
        probability = _prequential_predictions(train, predictor)
        metrics = probability_metrics(target, probability)
        score = float(metrics["brier"] + 0.15 * metrics["log_loss"])
        if best is None or score < best[0]:
            best = (score, spec)
    if best is None:
        raise ValueError(f"no candidate configuration for {name}")
    return best[1]


def _baseline_grids() -> dict[str, list[BaselineSpec]]:
    grids: dict[str, list[BaselineSpec]] = {"elo": [], "irt_2pl": [], "bkt": [], "pfa": []}
    for k_user in (0.12, 0.20, 0.32):
        k_item = 0.40 * k_user
        grids["elo"].append(BaselineSpec(
            "Elo",
            "elo",
            lambda ku=k_user, ki=k_item: OnlineElo(ku, ki),
            {"k_user": k_user, "k_item": k_item},
        ))
    for learning_rate in (0.018, 0.030, 0.045, 0.065):
        grids["irt_2pl"].append(BaselineSpec(
            "IRT-2PL online",
            "irt_2pl",
            lambda lr=learning_rate: OnlineIRT2PL(lr),
            {"learning_rate": learning_rate},
        ))
    for p_init in (0.20, 0.30):
        for p_learn in (0.08, 0.14, 0.20):
            for p_guess in (0.15, 0.22):
                for p_slip in (0.08, 0.14):
                    grids["bkt"].append(BaselineSpec(
                        "BKT",
                        "bkt",
                        lambda pi=p_init, pl=p_learn, pg=p_guess, ps=p_slip: OnlineBKT(pi, pl, pg, ps),
                        {"p_init": p_init, "p_learn": p_learn, "p_guess": p_guess, "p_slip": p_slip},
                    ))
    for intercept in (-0.75, -0.35, 0.0):
        for success_weight in (0.28, 0.42, 0.58):
            for failure_weight in (-0.62, -0.46, -0.30):
                for difficulty_weight in (1.5, 2.25):
                    grids["pfa"].append(BaselineSpec(
                        "PFA",
                        "pfa",
                        lambda a=intercept, s=success_weight, f=failure_weight, d=difficulty_weight: OnlinePFA(a, s, f, d),
                        {
                            "intercept": intercept,
                            "success_weight": success_weight,
                            "failure_weight": failure_weight,
                            "difficulty_weight": difficulty_weight,
                        },
                    ))
    return grids


def _rebuild_selected_ml(metadata: dict[str, Any], train: pd.DataFrame):
    best_name = str(metadata.get("best_model") or "")
    specs = [spec for spec in experiment_specs(get_settings().random_seed) if spec.name == best_name]
    if not specs:
        raise ValueError(f"active model specification is not reproducible: {best_name}")
    estimator = candidate_estimators(get_settings().random_seed, specs)[best_name]
    estimator.fit(train[NUMERIC_FEATURES + CATEGORICAL_FEATURES], train[TARGET_COLUMN].astype(int).to_numpy())
    return estimator


def _user_brier(frame: pd.DataFrame, probability: np.ndarray) -> dict[int, float]:
    ordered = _ordered(frame).reset_index(drop=True)
    target = ordered[TARGET_COLUMN].astype(int).to_numpy()
    loss = (np.asarray(probability, dtype=float) - target) ** 2
    result: dict[int, float] = {}
    for user_id, indices in ordered.groupby("user_id", sort=False).groups.items():
        idx = np.asarray(list(indices), dtype=int)
        result[int(user_id)] = float(np.mean(loss[idx]))
    return result


def benchmark_classical_baselines(frame: pd.DataFrame, metadata: dict[str, Any]) -> dict[str, Any]:
    validate_feature_frame(frame, require_target=True)
    split = metadata.get("split", {})
    train_users = {int(value) for value in split.get("train_users", [])}
    validation_users = {int(value) for value in split.get("validation_users", [])}
    if not train_users or not validation_users:
        raise ValueError("group-holdout user manifest is required for the baseline benchmark")
    if train_users.intersection(validation_users):
        raise ValueError("train and validation users overlap")

    train = _ordered(frame[frame["user_id"].isin(train_users)].copy())
    validation = _ordered(frame[frame["user_id"].isin(validation_users)].copy())
    if train.empty or validation.empty:
        raise ValueError("baseline benchmark split is empty")

    ml_estimator = _rebuild_selected_ml(metadata, train)
    x_validation = validation[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    target = validation[TARGET_COLUMN].astype(int).to_numpy()
    ml_probability = ml_estimator.predict_proba(x_validation)[:, 1]
    ml_metrics = probability_metrics(target, ml_probability)

    grids = _baseline_grids()
    rows: list[dict[str, Any]] = []
    prediction_by_name: dict[str, np.ndarray] = {"ML adaptive predictor": ml_probability}
    user_brier_by_name: dict[str, dict[int, float]] = {
        "ML adaptive predictor": _user_brier(validation, ml_probability),
    }

    for family in ("elo", "irt_2pl", "bkt", "pfa"):
        best = _tune(train, family, grids[family])
        predictor = best.factory()
        if isinstance(predictor, OnlineElo):
            _prequential_predictions(train, predictor)
            for key in list(predictor.ability):
                del predictor.ability[key]
            probability = _prequential_predictions(validation, predictor, freeze_global_items=True)
        else:
            probability = _prequential_predictions(validation, predictor)
        metrics = probability_metrics(target, probability)
        rows.append({
            "name": best.name,
            "family": family,
            "parameters": best.parameters,
            "metrics": metrics,
        })
        prediction_by_name[best.name] = probability
        user_brier_by_name[best.name] = _user_brier(validation, probability)

    comparisons: list[dict[str, Any]] = []
    ml_user_loss = user_brier_by_name["ML adaptive predictor"]
    for row in rows:
        baseline_name = row["name"]
        baseline_loss = user_brier_by_name[baseline_name]
        common_users = sorted(set(ml_user_loss).intersection(baseline_loss))
        left = [ml_user_loss[user_id] for user_id in common_users]
        right = [baseline_loss[user_id] for user_id in common_users]
        interval = bootstrap_difference_interval(left, right, iterations=2500, random_seed=42)
        p_value_one_sided = paired_randomization_pvalue(left, right, alternative="less", random_seed=42)
        p_value_two_sided = paired_randomization_pvalue(left, right, alternative="two-sided", random_seed=42)
        effect = paired_standardized_effect(left, right)
        comparisons.append({
            "baseline": baseline_name,
            "paired_users": len(common_users),
            "mean_user_brier_difference_ml_minus_baseline": interval.as_dict(),
            "p_value_one_sided_ml_lower": p_value_one_sided,
            "p_value_two_sided": p_value_two_sided,
            "paired_effect_dz": effect,
        })

    adjusted = holm_adjusted_pvalues(row["p_value_two_sided"] for row in comparisons)
    for row, adjusted_p in zip(comparisons, adjusted):
        row["p_value_holm_two_sided"] = adjusted_p
        row["ml_significantly_lower_brier"] = bool(
            row["mean_user_brier_difference_ml_minus_baseline"]["upper"] < 0.0 and adjusted_p < 0.05
        )

    all_rows = [{
        "name": "ML adaptive predictor",
        "family": str(metadata.get("best_family") or "ml"),
        "parameters": metadata.get("best_parameters", {}),
        "metrics": ml_metrics,
    }, *rows]
    all_rows.sort(key=lambda item: (item["metrics"]["brier"], -item["metrics"]["roc_auc"]))

    return {
        "research_question": "Does the ML predictor improve held-out probability quality over classical educational baselines?",
        "data_status": "synthetic bootstrap interaction log",
        "split": {
            "strategy": split.get("strategy"),
            "train_rows": int(len(train)),
            "validation_rows": int(len(validation)),
            "train_users": len(train_users),
            "validation_users": len(validation_users),
            "user_overlap": 0,
        },
        "primary_predictive_metric": "Brier score (lower is better)",
        "models": all_rows,
        "ml_vs_classical": comparisons,
        "best_by_brier": all_rows[0]["name"],
        "method_notes": [
            "The ML candidate is rebuilt on the original training-user split and evaluated on the original held-out users.",
            "IRT, BKT and PFA update only a learner's past state before predicting the next attempt; the current outcome is never used in its own prediction.",
            "Elo item ratings are learned on training users and frozen during held-out evaluation; only the held-out learner state adapts online.",
            "Baseline hyperparameters are selected on the training-user partition only.",
            "Paired significance tests use per-user mean Brier loss to avoid treating repeated attempts from one learner as independent subjects.",
            "Two-sided paired randomization p-values are adjusted across the four classical-baseline comparisons with the Holm procedure; one-sided values are retained only as descriptive directional statistics.",
        ],
    }
