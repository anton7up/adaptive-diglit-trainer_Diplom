from __future__ import annotations

from datetime import datetime
from typing import Any

from .. import models
from ..timeutils import utcnow_naive


def build_model_card(record: models.MLModelVersion, metadata: dict[str, Any], quality: dict[str, Any] | None = None) -> dict[str, Any]:
    winning = metadata.get("candidates", [{}])[0] if metadata.get("candidates") else {}
    validation = winning.get("validation", {})
    limitations = [
        "Prediction is probabilistic and must not be interpreted as a formal grade.",
        "Cold-start quality is limited for users with no prior attempts.",
        "Synthetic bootstrap observations are suitable for software demonstration only and must not be presented as empirical student research data.",
        "Model quality should be re-evaluated after distribution shifts, task-bank changes or accumulation of new real observations.",
    ]
    intended_uses = [
        "estimate probability of a correct response for a candidate task",
        "support adaptive task ranking near the configured learning-zone target",
        "provide aggregate calibration diagnostics for a teacher",
    ]
    prohibited_uses = [
        "automatic disciplinary decisions",
        "high-stakes admission or employment decisions",
        "replacement of a formal final examination",
    ]
    return {
        "model_details": {
            "version": record.version,
            "model_name": record.model_name,
            "training_run_id": record.training_run_id,
            "trained_at": record.trained_at.isoformat() + "Z" if record.trained_at else None,
            "dataset_hash": record.dataset_hash,
            "feature_count": len(record.feature_names or []),
        },
        "intended_use": intended_uses,
        "prohibited_use": prohibited_uses,
        "training_data": {
            "rows": metadata.get("dataset_rows"),
            "users": metadata.get("users"),
            "positive_rate": metadata.get("positive_rate"),
            "quality_gate": quality,
        },
        "evaluation": {
            "selection_metric": metadata.get("selection_metric"),
            "winning_configuration": winning.get("name"),
            "composite_score": winning.get("composite_score"),
            "validation": validation,
            "cross_validation_mean": winning.get("cross_validation_mean", {}),
            "experiment_count": metadata.get("experiment_count", len(metadata.get("candidates", []))),
        },
        "features": record.feature_names or [],
        "limitations": limitations,
        "monitoring": {
            "recommended": [
                "Brier score on recent labeled attempts",
                "calibration gap",
                "Population Stability Index for numeric features",
                "new observations since training",
            ],
            "retraining_policy": "trigger on data volume, model age, calibration or Brier degradation",
        },
        "generated_at": utcnow_naive().isoformat() + "Z",
    }
