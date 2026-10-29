from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    family: str
    scale_numeric: bool
    estimator: Any
    parameters: dict[str, Any]


def experiment_specs(random_seed: int) -> list[ExperimentSpec]:
    specs: list[ExperimentSpec] = []

    for c in (0.5, 2.0):
        params = {"C": c, "max_iter": 1200, "class_weight": "balanced"}
        specs.append(ExperimentSpec(
            name=f"logreg_c{str(c).replace('.', '_')}",
            family="logistic_regression",
            scale_numeric=True,
            estimator=LogisticRegression(
                C=c,
                max_iter=1200,
                class_weight="balanced",
                random_state=random_seed,
            ),
            parameters=params,
        ))

    forest_variants = [
        ("rf_regularized", 60, 9, 4, "sqrt"),
        ("rf_balanced", 90, 14, 3, 0.75),
    ]
    for name, trees, depth, leaf, max_features in forest_variants:
        params = {
            "n_estimators": trees,
            "max_depth": depth,
            "min_samples_leaf": leaf,
            "max_features": max_features,
        }
        specs.append(ExperimentSpec(
            name=name,
            family="random_forest",
            scale_numeric=False,
            estimator=RandomForestClassifier(
                n_estimators=trees,
                max_depth=depth,
                min_samples_leaf=leaf,
                max_features=max_features,
                class_weight="balanced_subsample",
                random_state=random_seed,
                n_jobs=-1,
            ),
            parameters=params,
        ))

    extra_variants = [
        ("extra_regularized", 70, 11, 4, "sqrt"),
        ("extra_balanced", 100, 16, 3, 0.85),
    ]
    for name, trees, depth, leaf, max_features in extra_variants:
        params = {
            "n_estimators": trees,
            "max_depth": depth,
            "min_samples_leaf": leaf,
            "max_features": max_features,
        }
        specs.append(ExperimentSpec(
            name=name,
            family="extra_trees",
            scale_numeric=False,
            estimator=ExtraTreesClassifier(
                n_estimators=trees,
                max_depth=depth,
                min_samples_leaf=leaf,
                max_features=max_features,
                class_weight="balanced",
                random_state=random_seed,
                n_jobs=-1,
            ),
            parameters=params,
        ))

    boosting_variants = [
        ("hgb_regularized", 80, 0.045, 15, 0.8),
        ("hgb_standard", 120, 0.055, 31, 0.3),
    ]
    for name, max_iter, learning_rate, max_leaf_nodes, l2_regularization in boosting_variants:
        params = {
            "max_iter": max_iter,
            "learning_rate": learning_rate,
            "max_leaf_nodes": max_leaf_nodes,
            "l2_regularization": l2_regularization,
        }
        specs.append(ExperimentSpec(
            name=name,
            family="hist_gradient_boosting",
            scale_numeric=False,
            estimator=HistGradientBoostingClassifier(
                max_iter=max_iter,
                learning_rate=learning_rate,
                max_leaf_nodes=max_leaf_nodes,
                l2_regularization=l2_regularization,
                random_state=random_seed,
            ),
            parameters=params,
        ))

    return specs
