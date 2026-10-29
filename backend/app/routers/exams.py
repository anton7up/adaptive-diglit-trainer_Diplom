from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session, joinedload

from .. import models, schemas
from ..database import get_db
from ..dependencies import get_current_user, require_roles
from ..services.exam_service import answer_exam_task, exam_payload, start_exam, submit_exam

router = APIRouter(prefix="/exams", tags=["exams"])


@router.get("", response_model=list[schemas.ExamSummary])
def list_exams(db: Session = Depends(get_db), _: models.User = Depends(get_current_user)):
    return db.query(models.Exam).filter(models.Exam.active.is_(True)).order_by(models.Exam.created_at.desc()).all()


@router.post("", response_model=schemas.ExamSummary, status_code=201)
def create_exam(
    payload: schemas.ExamCreate,
    db: Session = Depends(get_db),
    user: models.User = Depends(require_roles("teacher", "admin")),
):
    unique_ids = list(dict.fromkeys(payload.task_ids))
    tasks = db.query(models.Task).filter(models.Task.id.in_(unique_ids), models.Task.active.is_(True)).all()
    if len(tasks) != len(unique_ids):
        raise HTTPException(status_code=400, detail="One or more tasks do not exist")
    exam = models.Exam(
        title=payload.title,
        description=payload.description,
        duration_minutes=payload.duration_minutes,
        passing_score=payload.passing_score,
        created_by=user.id,
    )
    db.add(exam)
    db.flush()
    task_map = {task.id: task for task in tasks}
    for position, task_id in enumerate(unique_ids, start=1):
        db.add(models.ExamTask(exam_id=exam.id, task_id=task_id, position=position, points=1.0))
    db.commit()
    db.refresh(exam)
    return exam


@router.post("/{exam_id}/start", response_model=schemas.ExamStartResponse)
def start(
    exam_id: int,
    db: Session = Depends(get_db),
    user: models.User = Depends(get_current_user),
):
    exam = (
        db.query(models.Exam)
        .options(joinedload(models.Exam.tasks).joinedload(models.ExamTask.task))
        .filter(models.Exam.id == exam_id, models.Exam.active.is_(True))
        .first()
    )
    if exam is None:
        raise HTTPException(status_code=404, detail="Exam not found")
    session = start_exam(db, exam, user)
    return exam_payload(exam, session)


@router.post("/sessions/{session_id}/answers", status_code=204)
def answer(
    session_id: int,
    payload: schemas.ExamAnswerRequest,
    db: Session = Depends(get_db),
    user: models.User = Depends(get_current_user),
):
    session = (
        db.query(models.ExamSession)
        .options(joinedload(models.ExamSession.exam).joinedload(models.Exam.tasks).joinedload(models.ExamTask.task))
        .filter(models.ExamSession.id == session_id, models.ExamSession.user_id == user.id)
        .first()
    )
    if session is None:
        raise HTTPException(status_code=404, detail="Exam session not found")
    try:
        answer_exam_task(db, session, payload.task_id, payload.selected_option, payload.response_time_ms)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    return None


@router.post("/sessions/{session_id}/submit", response_model=schemas.ExamSubmitResponse)
def submit(
    session_id: int,
    db: Session = Depends(get_db),
    user: models.User = Depends(get_current_user),
):
    session = (
        db.query(models.ExamSession)
        .options(joinedload(models.ExamSession.answers), joinedload(models.ExamSession.exam).joinedload(models.Exam.tasks))
        .filter(models.ExamSession.id == session_id, models.ExamSession.user_id == user.id)
        .first()
    )
    if session is None:
        raise HTTPException(status_code=404, detail="Exam session not found")
    try:
        return submit_exam(db, session)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
