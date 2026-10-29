from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from .. import models, schemas
from ..database import get_db
from ..dependencies import get_current_user, require_roles
from ..services.analytics_service import student_progress, teacher_dashboard

router = APIRouter(tags=["analytics"])


@router.get("/progress/me", response_model=schemas.ProgressResponse)
def progress(db: Session = Depends(get_db), user: models.User = Depends(get_current_user)):
    return student_progress(db, user)


@router.get("/teacher/dashboard", response_model=schemas.TeacherDashboardResponse)
def dashboard(
    db: Session = Depends(get_db),
    _: models.User = Depends(require_roles("teacher", "admin")),
):
    return teacher_dashboard(db)
