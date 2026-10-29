from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from .. import models, schemas
from ..database import get_db
from ..dependencies import require_roles
from ..services.model_registry import ModelRegistry
from ..services.feature_engineering import build_training_frame
from ..services.data_quality import inspect_training_frame
from ..services.ablation import run_ablation
from ..services.retraining_policy import evaluate_retraining_need
from ..services.seed import seed_synthetic_history
from ..services.training_orchestrator import train_from_database

router = APIRouter(prefix="/ml", tags=["machine-learning"])


@router.get("/status", response_model=schemas.ModelStatusResponse)
def status(
    db: Session = Depends(get_db),
    _: models.User = Depends(require_roles("teacher", "admin")),
):
    row = ModelRegistry(db).active_record()
    if row is None:
        return schemas.ModelStatusResponse(active=False)
    return schemas.ModelStatusResponse(
        active=True,
        version=row.version,
        model_name=row.model_name,
        trained_at=row.trained_at,
        dataset_hash=row.dataset_hash,
        metrics=row.metrics,
        feature_names=row.feature_names,
    )


@router.get("/models")
def models_list(
    db: Session = Depends(get_db),
    _: models.User = Depends(require_roles("teacher", "admin")),
):
    rows = db.query(models.MLModelVersion).order_by(models.MLModelVersion.trained_at.desc()).all()
    return [
        {
            "version": row.version,
            "model_name": row.model_name,
            "is_active": row.is_active,
            "trained_at": row.trained_at,
            "metrics": row.metrics,
            "dataset_hash": row.dataset_hash,
        }
        for row in rows
    ]


@router.post("/models/{version}/activate")
def activate_model(
    version: str,
    db: Session = Depends(get_db),
    _: models.User = Depends(require_roles("admin")),
):
    try:
        row = ModelRegistry(db).activate(version)
        return {"version": row.version, "active": row.is_active}
    except ValueError:
        raise HTTPException(status_code=404, detail="Model version not found")


@router.post("/train", response_model=schemas.TrainingResponse)
def train(
    db: Session = Depends(get_db),
    _: models.User = Depends(require_roles("teacher", "admin")),
):
    try:
        result = train_from_database(db, activate=True)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    meta = result["metadata"]
    return schemas.TrainingResponse(
        run_id=result["run_id"],
        version=result["version"],
        model_name=result["model_name"],
        dataset_rows=meta["dataset_rows"],
        positive_rate=meta["positive_rate"],
        candidates=meta["candidates"],
    )


@router.post("/bootstrap", response_model=schemas.SeedSyntheticResponse)
def bootstrap(
    payload: schemas.SeedSyntheticRequest,
    db: Session = Depends(get_db),
    _: models.User = Depends(require_roles("admin")),
):
    created = seed_synthetic_history(db, payload.users, payload.attempts_per_user)
    return schemas.SeedSyntheticResponse(attempts_created=created)


@router.get("/audit", response_model=list[schemas.PredictionAuditResponse])
def audit(
    limit: int = 100,
    db: Session = Depends(get_db),
    _: models.User = Depends(require_roles("teacher", "admin")),
):
    limit = max(1, min(limit, 500))
    return db.query(models.PredictionAudit).order_by(models.PredictionAudit.created_at.desc()).limit(limit).all()


@router.get("/quality")
def quality(
    db: Session = Depends(get_db),
    _: models.User = Depends(require_roles("teacher", "admin")),
):
    attempts = db.query(models.Attempt).order_by(models.Attempt.created_at.asc(), models.Attempt.id.asc()).all()
    frame = build_training_frame(attempts)
    return inspect_training_frame(frame).as_dict()


@router.get("/retraining-policy")
def retraining_policy(
    db: Session = Depends(get_db),
    _: models.User = Depends(require_roles("teacher", "admin")),
):
    return evaluate_retraining_need(db).as_dict()


@router.post("/ablation")
def ablation(
    db: Session = Depends(get_db),
    _: models.User = Depends(require_roles("teacher", "admin")),
):
    attempts = db.query(models.Attempt).order_by(models.Attempt.created_at.asc(), models.Attempt.id.asc()).all()
    frame = build_training_frame(attempts)
    if len(frame) < 250 or frame["user_id"].nunique() < 5:
        raise HTTPException(status_code=422, detail="Not enough data for ablation study")
    return run_ablation(frame)
