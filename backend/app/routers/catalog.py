from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from .. import models, schemas
from ..database import get_db
from ..dependencies import get_current_user, require_roles

router = APIRouter(tags=["catalog"])


@router.get("/skills", response_model=list[schemas.SkillResponse])
def skills(db: Session = Depends(get_db), _: models.User = Depends(get_current_user)):
    return db.query(models.Skill).filter(models.Skill.active.is_(True)).order_by(models.Skill.sort_order).all()


@router.get("/tasks", response_model=list[schemas.TaskAdminResponse])
def tasks(
    skill_id: int | None = None,
    active: bool | None = None,
    db: Session = Depends(get_db),
    _: models.User = Depends(require_roles("teacher", "admin")),
):
    query = db.query(models.Task)
    if skill_id is not None:
        query = query.filter(models.Task.skill_id == skill_id)
    if active is not None:
        query = query.filter(models.Task.active == active)
    return query.order_by(models.Task.skill_id, models.Task.difficulty).all()


@router.post("/tasks", response_model=schemas.TaskAdminResponse, status_code=201)
def create_task(
    payload: schemas.TaskCreate,
    db: Session = Depends(get_db),
    _: models.User = Depends(require_roles("teacher", "admin")),
):
    if db.get(models.Skill, payload.skill_id) is None:
        raise HTTPException(status_code=404, detail="Skill not found")
    task = models.Task(**payload.model_dump())
    db.add(task)
    db.commit()
    db.refresh(task)
    return task


@router.patch("/tasks/{task_id}", response_model=schemas.TaskAdminResponse)
def update_task(
    task_id: int,
    payload: schemas.TaskUpdate,
    db: Session = Depends(get_db),
    _: models.User = Depends(require_roles("teacher", "admin")),
):
    task = db.get(models.Task, task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    for key, value in payload.model_dump(exclude_unset=True).items():
        setattr(task, key, value)
    db.commit()
    db.refresh(task)
    return task
