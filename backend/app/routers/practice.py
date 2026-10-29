from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from .. import models, schemas
from ..database import get_db
from ..dependencies import get_current_user
from ..services.adaptive_engine import choose_next_task
from ..services.inference import predict_tasks
from ..services.snapshot_service import refresh_user_skill_snapshot
from ..services.session_planner import build_practice_plan

router = APIRouter(prefix="/practice", tags=["practice"])


@router.post("/next", response_model=schemas.AdaptiveTaskResponse)
def next_task(
    payload: schemas.NextTaskRequest,
    db: Session = Depends(get_db),
    user: models.User = Depends(get_current_user),
):
    if user.role != "student":
        raise HTTPException(status_code=403, detail="Student role required")
    if payload.skill_id is not None and db.get(models.Skill, payload.skill_id) is None:
        raise HTTPException(status_code=404, detail="Skill not found")
    selection = choose_next_task(db, user.id, payload.skill_id)
    if selection is None:
        raise HTTPException(status_code=404, detail="No active tasks")
    task = selection.prediction.task
    return schemas.AdaptiveTaskResponse(
        id=task.id,
        skill_id=task.skill_id,
        skill_name=task.skill.name,
        question=task.question,
        options={"A": task.option_a, "B": task.option_b, "C": task.option_c, "D": task.option_d},
        difficulty=task.difficulty,
        cognitive_level=task.cognitive_level,
        estimated_time_sec=task.estimated_time_sec,
        predicted_probability=selection.prediction.probability,
        model_version=selection.prediction.model_version,
        selection_score=selection.score,
        selection_components=selection.components,
    )


@router.post("/submit", response_model=schemas.SubmitAnswerResponse)
def submit(
    payload: schemas.SubmitAnswerRequest,
    db: Session = Depends(get_db),
    user: models.User = Depends(get_current_user),
):
    if user.role != "student":
        raise HTTPException(status_code=403, detail="Student role required")
    task = db.get(models.Task, payload.task_id)
    if task is None or not task.active:
        raise HTTPException(status_code=404, detail="Task not found")
    prediction = predict_tasks(db, user.id, [task])[0]
    is_correct = payload.selected_option.upper() == task.correct_option
    attempt = models.Attempt(
        user_id=user.id,
        task_id=task.id,
        session_id=payload.session_id,
        selected_option=payload.selected_option.upper(),
        is_correct=is_correct,
        predicted_probability=prediction.probability,
        prediction_model=prediction.model_version,
        response_time_ms=payload.response_time_ms,
        hint_used=payload.hint_used,
        client_context={"source": "adaptive_practice"},
    )
    db.add(attempt)
    db.flush()
    refresh_user_skill_snapshot(db, user.id, task.skill_id)
    db.commit()
    db.refresh(attempt)
    return schemas.SubmitAnswerResponse(
        attempt_id=attempt.id,
        is_correct=is_correct,
        correct_option=task.correct_option,
        explanation=task.explanation,
        predicted_probability=attempt.predicted_probability,
        model_version=attempt.prediction_model,
    )


@router.get("/plan")
def practice_plan(
    length: int = 8,
    skill_id: int | None = None,
    db: Session = Depends(get_db),
    user: models.User = Depends(get_current_user),
):
    if user.role != "student":
        raise HTTPException(status_code=403, detail="Student role required")
    return build_practice_plan(db, user.id, length=length, skill_id=skill_id)
