from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session, joinedload

from .. import models
from ..database import get_db
from ..dependencies import require_roles
from ..services.classical_baselines import benchmark_classical_baselines
from ..services.cohort_analysis import attach_group_labels, evaluate_cohorts
from ..services.data_quality import inspect_training_frame
from ..services.dataset_profile import build_dataset_profile
from ..services.error_analysis import build_error_analysis
from ..services.experiment_analysis import repeated_group_benchmark
from ..services.explainability import explain_row, global_sensitivity, reference_profile
from ..services.feature_engineering import build_candidate_frame, build_training_frame
from ..services.learning_curve import learning_curve_report, summarize_learning_curve
from ..services.model_card import build_model_card
from ..services.model_evaluation import build_evaluation_report
from ..services.model_experiments import experiment_specs
from ..services.model_registry import ModelRegistry
from ..services.model_training import candidate_estimators
from ..services.monitoring_service import model_monitoring_report
from ..services.policy_simulation import run_policy_robustness, run_policy_simulation
from ..services.personalization_diagnostics import personalization_report
from ..services.robustness_analysis import robustness_report
from ..services.temporal_backtest import expanding_window_backtest
from ..services.validation_protocol import build_validation_protocol

router = APIRouter(prefix="/research", tags=["research"])


def _attempt_frame(db: Session):
    attempts = (
        db.query(models.Attempt)
        .options(joinedload(models.Attempt.task).joinedload(models.Task.skill))
        .order_by(models.Attempt.created_at.asc(), models.Attempt.id.asc())
        .all()
    )
    return build_training_frame(attempts)


def _context(db: Session):
    registry = ModelRegistry(db)
    record, estimator = registry.load_active()
    if record is None or estimator is None:
        raise HTTPException(status_code=404, detail="No active model")
    frame = _attempt_frame(db)
    if len(frame) < 100:
        raise HTTPException(status_code=422, detail="Not enough observations")
    return registry, record, estimator, frame


@router.get("/model-card")
def model_card(
    db: Session = Depends(get_db),
    _: models.User = Depends(require_roles("teacher", "admin")),
):
    registry, record, _, frame = _context(db)
    metadata = registry.metadata(record.version)
    quality = inspect_training_frame(frame).as_dict()
    return build_model_card(record, metadata, quality)


@router.get("/evaluation")
def evaluation(
    db: Session = Depends(get_db),
    _: models.User = Depends(require_roles("teacher", "admin")),
):
    _, record, estimator, frame = _context(db)
    return build_evaluation_report(estimator, frame, record.version)


@router.get("/errors")
def errors(
    db: Session = Depends(get_db),
    _: models.User = Depends(require_roles("teacher", "admin")),
):
    _, _, estimator, frame = _context(db)
    return build_error_analysis(estimator, frame)


@router.get("/learning-curve")
def learning_curve(
    db: Session = Depends(get_db),
    _: models.User = Depends(require_roles("teacher", "admin")),
):
    _, _, estimator, frame = _context(db)
    return summarize_learning_curve(learning_curve_report(estimator, frame))


@router.get("/explainability/global")
def explainability_global(
    db: Session = Depends(get_db),
    _: models.User = Depends(require_roles("teacher", "admin")),
):
    _, record, estimator, frame = _context(db)
    return {
        "model_version": record.version,
        "method": "counterfactual sensitivity around median/mode reference values",
        "features": global_sensitivity(estimator, frame),
    }


@router.get("/explainability/user/{user_id}/task/{task_id}")
def explain_prediction(
    user_id: int,
    task_id: int,
    db: Session = Depends(get_db),
    _: models.User = Depends(require_roles("teacher", "admin")),
):
    _, record, estimator, frame = _context(db)
    user = db.get(models.User, user_id)
    task = db.get(models.Task, task_id)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    if task is None or not task.active:
        raise HTTPException(status_code=404, detail="Task not found")
    row = build_candidate_frame(db, user.id, [task])
    explanation = explain_row(estimator, row, reference_profile(frame), max_features=8)
    return {
        "model_version": record.version,
        "user_id": user_id,
        "task_id": task_id,
        "skill": task.skill.code,
        **explanation,
    }


@router.get("/cohorts")
def cohorts(
    min_rows: int = Query(default=30, ge=10, le=500),
    min_users: int = Query(default=2, ge=1, le=50),
    db: Session = Depends(get_db),
    _: models.User = Depends(require_roles("teacher", "admin")),
):
    _, record, estimator, frame = _context(db)
    enriched = attach_group_labels(db, frame)
    return {
        "model_version": record.version,
        **evaluate_cohorts(estimator, enriched, min_rows=min_rows, min_users=min_users),
    }


@router.get("/monitoring")
def monitoring(
    recent_days: int = Query(default=14, ge=3, le=90),
    db: Session = Depends(get_db),
    _: models.User = Depends(require_roles("teacher", "admin")),
):
    return model_monitoring_report(db, recent_days=recent_days)


@router.get("/validation-protocol")
def validation_protocol(
    db: Session = Depends(get_db),
    _: models.User = Depends(require_roles("teacher", "admin")),
):
    registry, record, _, frame = _context(db)
    metadata = registry.metadata(record.version)
    return {"model_version": record.version, **build_validation_protocol(frame, metadata)}


@router.get("/dataset-profile")
def dataset_profile(
    db: Session = Depends(get_db),
    _: models.User = Depends(require_roles("teacher", "admin")),
):
    _, record, _, frame = _context(db)
    return {"model_version": record.version, **build_dataset_profile(frame)}


@router.get("/robustness")
def robustness(
    db: Session = Depends(get_db),
    _: models.User = Depends(require_roles("teacher", "admin")),
):
    _, record, estimator, frame = _context(db)
    return {"model_version": record.version, **robustness_report(estimator, frame)}


@router.get("/personalization")
def personalization(
    max_users: int = Query(default=18, ge=2, le=50),
    minimum_attempts: int = Query(default=8, ge=1, le=100),
    top_k: int = Query(default=5, ge=2, le=12),
    db: Session = Depends(get_db),
    _: models.User = Depends(require_roles("teacher", "admin")),
):
    _, record, estimator, _ = _context(db)
    try:
        report = personalization_report(
            db,
            estimator,
            max_users=max_users,
            minimum_attempts=minimum_attempts,
            top_k=top_k,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return {"model_version": record.version, **report}


@router.post("/temporal-backtest")
def temporal_backtest(
    windows: int = Query(default=4, ge=2, le=8),
    db: Session = Depends(get_db),
    _: models.User = Depends(require_roles("teacher", "admin")),
):
    _, record, estimator, frame = _context(db)
    try:
        report = expanding_window_backtest(estimator, frame, windows=windows)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return {"model_version": record.version, **report}


@router.post("/benchmark")
def benchmark(
    full: bool = False,
    folds: int = Query(default=3, ge=2, le=5),
    db: Session = Depends(get_db),
    _: models.User = Depends(require_roles("teacher", "admin")),
):
    _, _, _, frame = _context(db)
    specs = experiment_specs(42)
    if not full:
        seen = set()
        representatives = []
        for spec in specs:
            if spec.family not in seen:
                representatives.append(spec)
                seen.add(spec.family)
        specs = representatives
    estimators = candidate_estimators(42, specs)
    return repeated_group_benchmark(frame, estimators, folds=folds, repetitions=1)


@router.post("/classical-baselines")
def classical_baselines(
    db: Session = Depends(get_db),
    _: models.User = Depends(require_roles("teacher", "admin")),
):
    registry, record, _, frame = _context(db)
    metadata = registry.metadata(record.version)
    try:
        report = benchmark_classical_baselines(frame, metadata)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return {"model_version": record.version, **report}


@router.post("/policy-simulation")
def policy_simulation(
    learners: int = Query(default=60, ge=10, le=160),
    steps: int = Query(default=90, ge=20, le=180),
    mastery_threshold: float = Query(default=0.70, ge=0.50, le=0.95),
    minimum_skill_mastery: float = Query(default=0.50, ge=0.0, le=0.95),
    db: Session = Depends(get_db),
    _: models.User = Depends(require_roles("teacher", "admin")),
):
    _, record, estimator, _ = _context(db)
    tasks = (
        db.query(models.Task)
        .options(joinedload(models.Task.skill))
        .filter(models.Task.active.is_(True))
        .all()
    )
    try:
        report = run_policy_simulation(
            tasks,
            estimator,
            learners=learners,
            steps=steps,
            mastery_threshold=mastery_threshold,
            minimum_skill_mastery=minimum_skill_mastery,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return {"model_version": record.version, **report}


@router.post("/policy-robustness")
def policy_robustness(
    learners: int = Query(default=24, ge=10, le=80),
    steps: int = Query(default=90, ge=20, le=180),
    mastery_threshold: float = Query(default=0.70, ge=0.50, le=0.95),
    minimum_skill_mastery: float = Query(default=0.50, ge=0.0, le=0.95),
    db: Session = Depends(get_db),
    _: models.User = Depends(require_roles("teacher", "admin")),
):
    _, record, estimator, _ = _context(db)
    tasks = (
        db.query(models.Task)
        .options(joinedload(models.Task.skill))
        .filter(models.Task.active.is_(True))
        .all()
    )
    try:
        report = run_policy_robustness(
            tasks,
            estimator,
            learners=learners,
            steps=steps,
            mastery_threshold=mastery_threshold,
            minimum_skill_mastery=minimum_skill_mastery,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return {"model_version": record.version, **report}
