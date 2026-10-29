from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from math import exp
from typing import Any

import numpy as np
import pandas as pd

from .. import models
from .feature_engineering import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from .statistics_utils import (
    bootstrap_confidence_interval,
    bootstrap_difference_interval,
    normalized_entropy,
    paired_randomization_pvalue,
    paired_standardized_effect,
)


@dataclass(frozen=True)
class SimulationTask:
    id: int
    skill_id: int
    skill_code: str
    difficulty: float
    discrimination: float
    cognitive_level: int
    estimated_time_sec: int


@dataclass
class SimulatedLearner:
    learner_id: int
    latent_mastery: dict[int, float]
    attempts_total: int = 0
    correct_total: int = 0
    recent_results: deque[int] = field(default_factory=lambda: deque(maxlen=10))
    per_skill_results: dict[int, deque[int]] = field(default_factory=lambda: defaultdict(lambda: deque(maxlen=10)))
    per_skill_attempts: dict[int, int] = field(default_factory=lambda: defaultdict(int))
    per_skill_correct: dict[int, int] = field(default_factory=lambda: defaultdict(int))
    per_task_attempts: dict[int, int] = field(default_factory=lambda: defaultdict(int))
    per_task_correct: dict[int, int] = field(default_factory=lambda: defaultdict(int))
    task_last_step: dict[int, int] = field(default_factory=dict)
    skill_last_step: dict[int, int] = field(default_factory=dict)
    last_step: int | None = None
    last_response_sec: float = 0.0
    response_times: list[float] = field(default_factory=list)
    skill_response_times: dict[int, list[float]] = field(default_factory=lambda: defaultdict(list))
    correct_streak: int = 0
    incorrect_streak: int = 0
    skill_correct_streak: dict[int, int] = field(default_factory=lambda: defaultdict(int))
    skill_incorrect_streak: dict[int, int] = field(default_factory=lambda: defaultdict(int))

    def empirical_ratio(self, correct: int, attempts: int) -> float:
        return float((correct + 1.0) / (attempts + 2.0))

    def estimated_skill_mastery(self, skill_id: int) -> float:
        attempts = self.per_skill_attempts[skill_id]
        correct = self.per_skill_correct[skill_id]
        base = self.empirical_ratio(correct, attempts)
        recent = self.per_skill_results[skill_id]
        recent_rate = float(sum(recent) / len(recent)) if recent else 0.5
        return 0.65 * base + 0.35 * recent_rate

    def feature_row(self, task: SimulationTask, step: int) -> dict[str, float | int | str]:
        user_accuracy = self.empirical_ratio(self.correct_total, self.attempts_total)
        user_last5 = _recent_accuracy(self.recent_results, 5)
        user_last10 = _recent_accuracy(self.recent_results, 10)
        skill_attempts = self.per_skill_attempts[task.skill_id]
        skill_correct = self.per_skill_correct[task.skill_id]
        skill_accuracy = self.empirical_ratio(skill_correct, skill_attempts)
        skill_last5 = _recent_accuracy(self.per_skill_results[task.skill_id], 5)
        skill_last10 = _recent_accuracy(self.per_skill_results[task.skill_id], 10)
        task_attempts = self.per_task_attempts[task.id]
        task_accuracy = self.empirical_ratio(self.per_task_correct[task.id], task_attempts)
        total = max(self.attempts_total, 1)
        mean_response = float(np.mean(self.response_times)) if self.response_times else 0.0
        std_response = float(np.std(self.response_times)) if len(self.response_times) > 1 else 0.0
        skill_times = self.skill_response_times[task.skill_id]
        skill_mean_response = float(np.mean(skill_times)) if skill_times else 0.0
        skill_last_response = float(skill_times[-1]) if skill_times else 0.0
        since_last = _step_seconds(step, self.last_step)
        since_skill = _step_seconds(step, self.skill_last_step.get(task.skill_id))
        since_task = _step_seconds(step, self.task_last_step.get(task.id))
        skill_mastery = self.estimated_skill_mastery(task.skill_id)
        speed_ratio = self.last_response_sec / max(task.estimated_time_sec, 1) if self.last_response_sec else 1.0
        hour = (9 + step // 4) % 24
        radians = 2.0 * np.pi * hour / 24.0
        return {
            "user_attempts_total": self.attempts_total,
            "user_accuracy_total": user_accuracy,
            "user_accuracy_last5": user_last5,
            "user_accuracy_last10": user_last10,
            "user_attempts_24h": min(self.attempts_total, 16),
            "user_attempts_7d": min(self.attempts_total, 64),
            "skill_attempts": skill_attempts,
            "skill_accuracy": skill_accuracy,
            "skill_accuracy_last5": skill_last5,
            "skill_accuracy_last10": skill_last10,
            "skill_attempts_7d": min(skill_attempts, 32),
            "task_attempts": task_attempts,
            "task_accuracy": task_accuracy,
            "task_exposure_share": task_attempts / total,
            "skill_exposure_share": skill_attempts / total,
            "seconds_since_last_attempt": since_last,
            "seconds_since_last_skill_attempt": since_skill,
            "seconds_since_last_task_attempt": since_task,
            "response_time_mean_sec": mean_response,
            "response_time_std_sec": std_response,
            "response_time_last_sec": self.last_response_sec,
            "skill_response_time_mean_sec": skill_mean_response,
            "skill_response_time_last_sec": skill_last_response,
            "correct_streak": self.correct_streak,
            "incorrect_streak": self.incorrect_streak,
            "skill_correct_streak": self.skill_correct_streak[task.skill_id],
            "skill_incorrect_streak": self.skill_incorrect_streak[task.skill_id],
            "task_difficulty": task.difficulty,
            "task_discrimination": task.discrimination,
            "task_cognitive_level": task.cognitive_level,
            "task_estimated_time_sec": task.estimated_time_sec,
            "difficulty_gap": task.difficulty - skill_mastery,
            "speed_ratio": float(min(max(speed_ratio, 0.0), 10.0)),
            "practice_density_7d": min(self.attempts_total, 64) / 7.0,
            "local_hour_sin": float(np.sin(radians)),
            "local_hour_cos": float(np.cos(radians)),
            "is_first_skill_attempt": int(skill_attempts == 0),
            "is_first_task_attempt": int(task_attempts == 0),
            "skill_code": task.skill_code,
        }

    def answer_probability(self, task: SimulationTask) -> float:
        mastery = self.latent_mastery[task.skill_id]
        logit = 5.0 * task.discrimination * (mastery - task.difficulty)
        return float(1.0 / (1.0 + exp(-max(min(logit, 20.0), -20.0))))

    def consume(self, task: SimulationTask, correct: bool, response_sec: float, step: int) -> None:
        y = int(correct)
        self.attempts_total += 1
        self.correct_total += y
        self.recent_results.append(y)
        self.per_skill_results[task.skill_id].append(y)
        self.per_skill_attempts[task.skill_id] += 1
        self.per_skill_correct[task.skill_id] += y
        self.per_task_attempts[task.id] += 1
        self.per_task_correct[task.id] += y
        self.task_last_step[task.id] = step
        self.skill_last_step[task.skill_id] = step
        self.last_step = step
        self.last_response_sec = response_sec
        self.response_times.append(response_sec)
        self.skill_response_times[task.skill_id].append(response_sec)
        if y:
            self.correct_streak += 1
            self.incorrect_streak = 0
            self.skill_correct_streak[task.skill_id] += 1
            self.skill_incorrect_streak[task.skill_id] = 0
        else:
            self.incorrect_streak += 1
            self.correct_streak = 0
            self.skill_incorrect_streak[task.skill_id] += 1
            self.skill_correct_streak[task.skill_id] = 0

        current = self.latent_mastery[task.skill_id]
        challenge = exp(-((task.difficulty - current) ** 2) / (2 * 0.23**2))
        learning_rate = 0.020 + 0.035 * challenge
        if correct:
            current += learning_rate * (1.0 - current)
        else:
            current += 0.45 * learning_rate * (1.0 - current)
        self.latent_mastery[task.skill_id] = float(np.clip(current, 0.02, 0.98))
        for skill_id in self.latent_mastery:
            if skill_id != task.skill_id:
                self.latent_mastery[skill_id] = float(max(0.02, self.latent_mastery[skill_id] - 0.00035))


def tasks_from_models(tasks: list[models.Task]) -> list[SimulationTask]:
    return [
        SimulationTask(
            id=task.id,
            skill_id=task.skill_id,
            skill_code=task.skill.code,
            difficulty=float(task.difficulty),
            discrimination=float(task.discrimination),
            cognitive_level=int(task.cognitive_level),
            estimated_time_sec=int(task.estimated_time_sec),
        )
        for task in tasks
        if task.active and task.skill is not None
    ]


def run_policy_simulation(
    tasks: list[models.Task],
    estimator,
    *,
    learners: int = 60,
    steps: int = 90,
    random_seed: int = 42,
    mastery_threshold: float = 0.70,
    minimum_skill_mastery: float = 0.50,
    inference_iterations: int = 20000,
) -> dict[str, Any]:
    items = tasks_from_models(tasks)
    if len(items) < 8:
        raise ValueError("simulation requires at least eight active tasks")
    skill_ids = sorted({task.skill_id for task in items})
    if len(skill_ids) < 2:
        raise ValueError("simulation requires tasks from at least two skills")
    if not 0.50 <= mastery_threshold <= 0.95:
        raise ValueError("mastery threshold must be between 0.50 and 0.95")
    if not 0.0 <= minimum_skill_mastery <= mastery_threshold:
        raise ValueError("minimum skill mastery must be between 0 and the mastery threshold")

    policies = ["random", "difficulty_match", "weak_skill", "ml_adaptive"]
    results: dict[str, list[dict[str, Any]]] = {name: [] for name in policies}
    trajectories: dict[str, list[list[float]]] = {name: [] for name in policies}
    base_rng = np.random.default_rng(random_seed)
    initial_profiles = [
        {skill_id: float(np.clip(base_rng.beta(3.0, 3.0), 0.08, 0.92)) for skill_id in skill_ids}
        for _ in range(learners)
    ]
    outcome_uniforms = base_rng.random((learners, steps))
    response_noise = base_rng.lognormal(mean=0.0, sigma=0.18, size=(learners, steps))
    selection_seeds = base_rng.integers(1, np.iinfo(np.int32).max, size=(len(policies), learners))

    for policy_index, policy in enumerate(policies):
        if policy == "ml_adaptive":
            states: list[dict[str, Any]] = []
            for learner_index, initial in enumerate(initial_profiles):
                learner = SimulatedLearner(learner_index, dict(initial))
                initial_mastery = float(np.mean(list(initial.values())))
                states.append({
                    "learner": learner,
                    "initial_mastery": initial_mastery,
                    "selected_skills": [],
                    "challenge_zone": 0,
                    "predicted": [],
                    "observed": [],
                    "trajectory": [initial_mastery],
                    "mastery_step": 0 if _mastery_reached(learner, mastery_threshold, minimum_skill_mastery) else None,
                })

            for step in range(steps):
                selections = _select_ml_batch([state["learner"] for state in states], items, estimator, step)
                for learner_index, (task, predicted_probability) in enumerate(selections):
                    state = states[learner_index]
                    learner = state["learner"]
                    true_probability = learner.answer_probability(task)
                    correct = bool(outcome_uniforms[learner_index, step] < true_probability)
                    response_sec = _response_time_from_noise(
                        task, learner, correct, float(response_noise[learner_index, step])
                    )
                    learner.consume(task, correct, response_sec, step)
                    state["selected_skills"].append(task.skill_id)
                    state["challenge_zone"] += int(0.45 <= true_probability <= 0.85)
                    state["predicted"].append(predicted_probability)
                    state["observed"].append(int(correct))
                    current_mastery = float(np.mean(list(learner.latent_mastery.values())))
                    state["trajectory"].append(current_mastery)
                    if state["mastery_step"] is None and _mastery_reached(learner, mastery_threshold, minimum_skill_mastery):
                        state["mastery_step"] = step + 1

            for learner_index, state in enumerate(states):
                learner = state["learner"]
                initial_mastery = state["initial_mastery"]
                final_mastery = state["trajectory"][-1]
                mastery_gain = final_mastery - initial_mastery
                normalized_mastery_gain = mastery_gain / max(1.0 - initial_mastery, 1e-9)
                mastery_step = state["mastery_step"]
                restricted_steps = float(mastery_step if mastery_step is not None else steps)
                predicted = np.asarray(state["predicted"], dtype=float)
                observed = np.asarray(state["observed"], dtype=float)
                selected_skills = state["selected_skills"]
                results[policy].append({
                    "learner_id": learner_index,
                    "initial_mastery": initial_mastery,
                    "final_mastery": final_mastery,
                    "mastery_gain": mastery_gain,
                    "normalized_mastery_gain": normalized_mastery_gain,
                    "reached_mastery": mastery_step is not None,
                    "steps_to_mastery": mastery_step,
                    "restricted_steps_to_mastery": restricted_steps,
                    "accuracy": learner.correct_total / max(learner.attempts_total, 1),
                    "challenge_zone_share": state["challenge_zone"] / steps,
                    "skill_coverage": len(set(selected_skills)) / len(skill_ids),
                    "skill_exposure_entropy": _skill_entropy(selected_skills, skill_ids),
                    "prediction_brier": float(np.mean((predicted - observed) ** 2)),
                })
                trajectories[policy].append(state["trajectory"])
            continue

        for learner_index, initial in enumerate(initial_profiles):
            learner = SimulatedLearner(learner_index, dict(initial))
            rng = np.random.default_rng(int(selection_seeds[policy_index, learner_index]))
            initial_mastery = float(np.mean(list(initial.values())))
            selected_skills: list[int] = []
            challenge_zone = 0
            predicted: list[float] = []
            observed: list[int] = []
            trajectory = [initial_mastery]
            mastery_step = 0 if _mastery_reached(learner, mastery_threshold, minimum_skill_mastery) else None

            for step in range(steps):
                task, predicted_probability = _select(policy, learner, items, estimator, step, rng)
                true_probability = learner.answer_probability(task)
                correct = bool(outcome_uniforms[learner_index, step] < true_probability)
                response_sec = _response_time_from_noise(
                    task,
                    learner,
                    correct,
                    float(response_noise[learner_index, step]),
                )
                learner.consume(task, correct, response_sec, step)
                selected_skills.append(task.skill_id)
                challenge_zone += int(0.45 <= true_probability <= 0.85)
                if predicted_probability is not None:
                    predicted.append(predicted_probability)
                    observed.append(int(correct))
                current_mastery = float(np.mean(list(learner.latent_mastery.values())))
                trajectory.append(current_mastery)
                if mastery_step is None and _mastery_reached(learner, mastery_threshold, minimum_skill_mastery):
                    mastery_step = step + 1

            final_mastery = trajectory[-1]
            mastery_gain = final_mastery - initial_mastery
            normalized_mastery_gain = mastery_gain / max(1.0 - initial_mastery, 1e-9)
            restricted_steps = float(mastery_step if mastery_step is not None else steps)
            results[policy].append({
                "learner_id": learner_index,
                "initial_mastery": initial_mastery,
                "final_mastery": final_mastery,
                "mastery_gain": mastery_gain,
                "normalized_mastery_gain": normalized_mastery_gain,
                "reached_mastery": mastery_step is not None,
                "steps_to_mastery": mastery_step,
                "restricted_steps_to_mastery": restricted_steps,
                "accuracy": learner.correct_total / max(learner.attempts_total, 1),
                "challenge_zone_share": challenge_zone / steps,
                "skill_coverage": len(set(selected_skills)) / len(skill_ids),
                "skill_exposure_entropy": _skill_entropy(selected_skills, skill_ids),
                "prediction_brier": float(np.mean((np.asarray(predicted) - np.asarray(observed)) ** 2)) if predicted else None,
            })
            trajectories[policy].append(trajectory)

    summaries = [_summarize_policy(name, rows) for name, rows in results.items()]
    summaries.sort(key=lambda item: item["mastery_gain"]["estimate"], reverse=True)
    endpoint = _mastery_endpoint(results, mastery_threshold, minimum_skill_mastery, steps, random_seed, inference_iterations)
    learning_gain_endpoint = _learning_gain_endpoint(results, random_seed, inference_iterations)

    return {
        "synthetic_experiment": True,
        "purpose": "controlled adaptive-policy comparison; not evidence from real students",
        "primary_outcome": "learning efficiency and learning gain under a fixed synthetic mastery model",
        "primary_endpoints": ["restricted mean tasks to mastery", "mastery gain at the fixed horizon"],
        "learners": learners,
        "steps_per_learner": steps,
        "tasks": len(items),
        "skills": len(skill_ids),
        "random_seed": random_seed,
        "mastery_definition": {
            "mean_latent_mastery_at_least": mastery_threshold,
            "every_skill_mastery_at_least": minimum_skill_mastery,
            "censoring_horizon_tasks": steps,
        },
        "primary_endpoint": endpoint,
        "learning_gain_endpoint": learning_gain_endpoint,
        "policies": summaries,
        "learning_trajectories": [
            _trajectory_summary(policy, trajectories[policy]) for policy in policies
        ],
        "winner_by_mean_mastery_gain": summaries[0]["policy"],
        "method_notes": [
            "All policies receive identical initial latent-mastery profiles.",
            "Matched learners also receive the same per-step random outcome uniforms and response-time noise, reducing Monte Carlo variance in paired comparisons.",
            "Outcomes are generated by a logistic item-response process with learning and mild forgetting.",
            "The ML policy uses only observable simulated history features available before the current response.",
            "The pedagogical endpoints are time-to-mastery and mastery gain; predictive AUC is treated as a supporting model-quality metric.",
            "Learners who do not reach mastery within the horizon are right-censored at the simulation horizon for the restricted-mean comparison.",
            "Synthetic results demonstrate algorithmic behavior and reproducibility, not pedagogical effectiveness in a human population.",
        ],
    }


def _mastery_reached(learner: SimulatedLearner, threshold: float, minimum_skill: float) -> bool:
    values = np.asarray(list(learner.latent_mastery.values()), dtype=float)
    return bool(values.size and float(np.mean(values)) >= threshold and float(np.min(values)) >= minimum_skill)


def _mastery_endpoint(
    results: dict[str, list[dict[str, Any]]],
    threshold: float,
    minimum_skill: float,
    horizon: int,
    random_seed: int,
    inference_iterations: int,
) -> dict[str, Any]:
    adaptive_rows = results["ml_adaptive"]
    random_rows = results["random"]
    adaptive_steps = [float(row["restricted_steps_to_mastery"]) for row in adaptive_rows]
    random_steps = [float(row["restricted_steps_to_mastery"]) for row in random_rows]
    reduction = bootstrap_difference_interval(
        random_steps, adaptive_steps, iterations=3000, random_seed=random_seed
    )
    p_value = paired_randomization_pvalue(
        random_steps, adaptive_steps, alternative="greater", iterations=inference_iterations, random_seed=random_seed
    )
    adaptive_mean = float(np.mean(adaptive_steps))
    random_mean = float(np.mean(random_steps))
    relative_reduction = (random_mean - adaptive_mean) / random_mean if random_mean > 0 else 0.0
    adaptive_reach = float(np.mean([row["reached_mastery"] for row in adaptive_rows]))
    random_reach = float(np.mean([row["reached_mastery"] for row in random_rows]))
    significant = bool(reduction.lower > 0.0 and p_value < 0.05)
    p_text = "p<0.001" if p_value < 0.001 else f"p={p_value:.3f}"
    claim = (
        f"In the controlled synthetic simulation, ML-adaptive selection reached the mastery criterion in "
        f"{adaptive_mean:.1f} tasks versus {random_mean:.1f} for random selection "
        f"(restricted mean; difference {reduction.estimate:.1f}, 95% CI {reduction.lower:.1f} to {reduction.upper:.1f}, "
        f"{p_text})."
    )
    return {
        "adaptive_policy": "ml_adaptive",
        "comparator": "random",
        "mastery_threshold": threshold,
        "minimum_skill_mastery": minimum_skill,
        "horizon_tasks": horizon,
        "adaptive_restricted_mean_tasks": adaptive_mean,
        "random_restricted_mean_tasks": random_mean,
        "tasks_saved": reduction.as_dict(),
        "relative_task_reduction": float(relative_reduction),
        "adaptive_reach_rate": adaptive_reach,
        "random_reach_rate": random_reach,
        "reach_rate_difference": adaptive_reach - random_reach,
        "p_value_one_sided_random_slower": p_value,
        "paired_effect_dz_random_minus_adaptive": paired_standardized_effect(random_steps, adaptive_steps),
        "statistically_significant": significant,
        "defense_claim": claim,
        "interpretation_guardrail": "This endpoint supports algorithmic adaptivity in simulation only; a randomized human pilot is still required for pedagogical effectiveness.",
    }


def _learning_gain_endpoint(
    results: dict[str, list[dict[str, Any]]], random_seed: int, inference_iterations: int
) -> dict[str, Any]:
    adaptive_rows = results["ml_adaptive"]
    random_rows = results["random"]
    adaptive_gain = [float(row["mastery_gain"]) for row in adaptive_rows]
    random_gain = [float(row["mastery_gain"]) for row in random_rows]
    adaptive_normalized = [float(row["normalized_mastery_gain"]) for row in adaptive_rows]
    random_normalized = [float(row["normalized_mastery_gain"]) for row in random_rows]
    difference = bootstrap_difference_interval(
        adaptive_gain, random_gain, iterations=3000, random_seed=random_seed + 101
    )
    normalized_difference = bootstrap_difference_interval(
        adaptive_normalized, random_normalized, iterations=3000, random_seed=random_seed + 103
    )
    p_value = paired_randomization_pvalue(
        adaptive_gain, random_gain, alternative="greater", iterations=inference_iterations, random_seed=random_seed + 107
    )
    return {
        "adaptive_mean_mastery_gain": float(np.mean(adaptive_gain)),
        "random_mean_mastery_gain": float(np.mean(random_gain)),
        "mastery_gain_advantage": difference.as_dict(),
        "adaptive_mean_normalized_gain": float(np.mean(adaptive_normalized)),
        "random_mean_normalized_gain": float(np.mean(random_normalized)),
        "normalized_gain_advantage": normalized_difference.as_dict(),
        "p_value_one_sided_adaptive_greater": p_value,
        "paired_effect_dz_adaptive_minus_random": paired_standardized_effect(adaptive_gain, random_gain),
        "statistically_significant": bool(difference.lower > 0.0 and p_value < 0.05),
        "interpretation_guardrail": "Mastery is latent and generated by the declared synthetic learner model; the endpoint is not a measured human learning gain.",
    }


def run_policy_robustness(
    tasks: list[models.Task],
    estimator,
    *,
    seeds: tuple[int, ...] = (11, 23, 37, 53, 71),
    learners: int = 24,
    steps: int = 90,
    mastery_threshold: float = 0.70,
    minimum_skill_mastery: float = 0.50,
) -> dict[str, Any]:
    if not seeds:
        raise ValueError("at least one simulation seed is required")
    runs: list[dict[str, Any]] = []
    for seed in seeds:
        report = run_policy_simulation(
            tasks,
            estimator,
            learners=learners,
            steps=steps,
            random_seed=int(seed),
            mastery_threshold=mastery_threshold,
            minimum_skill_mastery=minimum_skill_mastery,
            inference_iterations=3000,
        )
        endpoint = report["primary_endpoint"]
        gain = report["learning_gain_endpoint"]
        runs.append({
            "seed": int(seed),
            "adaptive_tasks": endpoint["adaptive_restricted_mean_tasks"],
            "random_tasks": endpoint["random_restricted_mean_tasks"],
            "tasks_saved": endpoint["tasks_saved"]["estimate"],
            "relative_task_reduction": endpoint["relative_task_reduction"],
            "time_to_mastery_p_value": endpoint["p_value_one_sided_random_slower"],
            "time_to_mastery_significant": endpoint["statistically_significant"],
            "mastery_gain_advantage": gain["mastery_gain_advantage"]["estimate"],
            "mastery_gain_p_value": gain["p_value_one_sided_adaptive_greater"],
            "mastery_gain_significant": gain["statistically_significant"],
        })
    saved = np.asarray([row["tasks_saved"] for row in runs], dtype=float)
    gains = np.asarray([row["mastery_gain_advantage"] for row in runs], dtype=float)
    return {
        "synthetic_experiment": True,
        "seeds": [int(value) for value in seeds],
        "learners_per_seed": int(learners),
        "steps_per_learner": int(steps),
        "runs": runs,
        "summary": {
            "positive_time_to_mastery_runs": int(np.sum(saved > 0.0)),
            "significant_time_to_mastery_runs": int(sum(row["time_to_mastery_significant"] for row in runs)),
            "mean_tasks_saved": float(np.mean(saved)),
            "min_tasks_saved": float(np.min(saved)),
            "max_tasks_saved": float(np.max(saved)),
            "positive_mastery_gain_runs": int(np.sum(gains > 0.0)),
            "significant_mastery_gain_runs": int(sum(row["mastery_gain_significant"] for row in runs)),
            "mean_mastery_gain_advantage": float(np.mean(gains)),
            "min_mastery_gain_advantage": float(np.min(gains)),
            "max_mastery_gain_advantage": float(np.max(gains)),
        },
        "interpretation_guardrail": "Seed robustness checks Monte Carlo stability of the declared synthetic experiment and does not replace validation on real learners.",
    }


def _trajectory_summary(policy: str, trajectories: list[list[float]]) -> dict[str, Any]:
    matrix = np.asarray(trajectories, dtype=float)
    mean = np.mean(matrix, axis=0)
    q25 = np.quantile(matrix, 0.25, axis=0)
    q75 = np.quantile(matrix, 0.75, axis=0)
    stride = max(1, (matrix.shape[1] - 1) // 30)
    indices = sorted(set(list(range(0, matrix.shape[1], stride)) + [matrix.shape[1] - 1]))
    return {
        "policy": policy,
        "points": [
            {
                "step": int(index),
                "mean_mastery": float(mean[index]),
                "q25": float(q25[index]),
                "q75": float(q75[index]),
            }
            for index in indices
        ],
    }


def _select_ml_batch(
    learners: list[SimulatedLearner],
    items: list[SimulationTask],
    estimator,
    step: int,
) -> list[tuple[SimulationTask, float]]:
    feature_rows: list[dict[str, float | int | str]] = []
    spans: list[tuple[list[SimulationTask], int, int]] = []
    for learner in learners:
        candidates = _novel_candidates(learner, items, step)
        start = len(feature_rows)
        feature_rows.extend(learner.feature_row(task, step) for task in candidates)
        spans.append((candidates, start, len(feature_rows)))

    frame = pd.DataFrame(feature_rows)
    probabilities = estimator.predict_proba(frame[NUMERIC_FEATURES + CATEGORICAL_FEATURES])[:, 1]
    selections: list[tuple[SimulationTask, float]] = []
    for learner, (candidates, start, end) in zip(learners, spans):
        local_probabilities = probabilities[start:end]
        best: tuple[float, SimulationTask, float] | None = None
        for task, probability in zip(candidates, local_probabilities):
            weakness = 1.0 - learner.estimated_skill_mastery(task.skill_id)
            novelty = 1.0 / (1.0 + learner.per_task_attempts[task.id])
            zone = exp(-((float(probability) - 0.70) ** 2) / (2 * 0.15**2))
            score = 0.60 * zone + 0.25 * weakness + 0.15 * novelty
            candidate = (score, task, float(probability))
            if best is None or candidate[0] > best[0]:
                best = candidate
        if best is None:
            raise ValueError("ML policy has no candidate tasks")
        selections.append((best[1], best[2]))
    return selections


def _select(policy: str, learner: SimulatedLearner, items: list[SimulationTask], estimator, step: int, rng):
    candidates = _novel_candidates(learner, items, step)
    if policy == "random":
        task = candidates[int(rng.integers(0, len(candidates)))]
        return task, None
    if policy == "difficulty_match":
        scored = [(abs(task.difficulty - learner.estimated_skill_mastery(task.skill_id)), task) for task in candidates]
        task = min(scored, key=lambda item: item[0])[1]
        return task, None
    if policy == "weak_skill":
        weakest = min({task.skill_id for task in candidates}, key=learner.estimated_skill_mastery)
        skill_tasks = [task for task in candidates if task.skill_id == weakest]
        task = min(skill_tasks, key=lambda item: abs(item.difficulty - learner.estimated_skill_mastery(weakest)))
        return task, None
    if policy != "ml_adaptive":
        raise ValueError(f"unknown policy: {policy}")
    frame = pd.DataFrame([learner.feature_row(task, step) for task in candidates])
    probabilities = estimator.predict_proba(frame[NUMERIC_FEATURES + CATEGORICAL_FEATURES])[:, 1]
    scores = []
    for task, probability in zip(candidates, probabilities):
        weakness = 1.0 - learner.estimated_skill_mastery(task.skill_id)
        novelty = 1.0 / (1.0 + learner.per_task_attempts[task.id])
        zone = exp(-((float(probability) - 0.70) ** 2) / (2 * 0.15**2))
        score = 0.60 * zone + 0.25 * weakness + 0.15 * novelty
        scores.append((score, task, float(probability)))
    _, task, probability = max(scores, key=lambda item: item[0])
    return task, probability


def _novel_candidates(learner: SimulatedLearner, items: list[SimulationTask], step: int) -> list[SimulationTask]:
    filtered = [task for task in items if step - learner.task_last_step.get(task.id, -999) > 2]
    return filtered or items


def _response_time_from_noise(
    task: SimulationTask,
    learner: SimulatedLearner,
    correct: bool,
    noise: float,
) -> float:
    mastery = learner.latent_mastery[task.skill_id]
    difficulty_load = 0.65 + 0.85 * max(task.difficulty - mastery + 0.35, 0.1)
    error_factor = 1.15 if not correct else 0.95
    return float(np.clip(task.estimated_time_sec * difficulty_load * error_factor * noise, 3.0, 300.0))


def _sample_response_time(task: SimulationTask, learner: SimulatedLearner, correct: bool, rng) -> float:
    return _response_time_from_noise(task, learner, correct, float(rng.lognormal(mean=0.0, sigma=0.18)))


def _recent_accuracy(values: deque[int], length: int) -> float:
    sample = list(values)[-length:]
    return float(sum(sample) / len(sample)) if sample else 0.5


def _step_seconds(step: int, previous: int | None) -> float:
    if previous is None:
        return 30.0 * 86400.0
    return float(max(step - previous, 0) * 90)


def _skill_entropy(selected: list[int], skill_ids: list[int]) -> float:
    counts = [selected.count(skill_id) for skill_id in skill_ids]
    return normalized_entropy(counts)


def _summarize_policy(name: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = ["mastery_gain", "normalized_mastery_gain", "final_mastery", "restricted_steps_to_mastery", "accuracy", "challenge_zone_share", "skill_coverage", "skill_exposure_entropy"]
    output = {"policy": name, "learners": len(rows)}
    for metric in metrics:
        output[metric] = bootstrap_confidence_interval([row[metric] for row in rows], iterations=700).as_dict()
    brier_values = [row["prediction_brier"] for row in rows if row["prediction_brier"] is not None]
    output["prediction_brier"] = bootstrap_confidence_interval(brier_values, iterations=700).as_dict() if brier_values else None
    output["mastery_reach_rate"] = float(np.mean([row["reached_mastery"] for row in rows]))
    reached_steps = [float(row["steps_to_mastery"]) for row in rows if row["steps_to_mastery"] is not None]
    output["steps_to_mastery_among_reached"] = (
        bootstrap_confidence_interval(reached_steps, iterations=700).as_dict() if reached_steps else None
    )
    return output
