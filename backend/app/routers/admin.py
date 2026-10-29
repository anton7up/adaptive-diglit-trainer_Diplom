from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from .. import models, schemas
from ..database import get_db
from ..dependencies import require_roles
from ..security import hash_password
from ..services.snapshot_service import refresh_all_snapshots

router = APIRouter(prefix="/admin", tags=["admin"])


@router.get("/users", response_model=list[schemas.UserResponse])
def list_users(
    role: str | None = None,
    active: bool | None = None,
    db: Session = Depends(get_db),
    _: models.User = Depends(require_roles("admin")),
):
    query = db.query(models.User)
    if role is not None:
        query = query.filter(models.User.role == role)
    if active is not None:
        query = query.filter(models.User.is_active == active)
    return query.order_by(models.User.created_at.desc()).all()


@router.patch("/users/{user_id}/role", response_model=schemas.UserResponse)
def update_role(
    user_id: int,
    role: str,
    db: Session = Depends(get_db),
    current: models.User = Depends(require_roles("admin")),
):
    if role not in {"student", "teacher", "admin"}:
        raise HTTPException(status_code=422, detail="Unsupported role")
    user = db.get(models.User, user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    if user.id == current.id and role != "admin":
        raise HTTPException(status_code=409, detail="Administrator cannot remove own admin role")
    user.role = role
    db.commit(); db.refresh(user)
    return user


@router.patch("/users/{user_id}/active", response_model=schemas.UserResponse)
def update_active(
    user_id: int,
    active: bool,
    db: Session = Depends(get_db),
    current: models.User = Depends(require_roles("admin")),
):
    user = db.get(models.User, user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    if user.id == current.id and not active:
        raise HTTPException(status_code=409, detail="Administrator cannot deactivate own account")
    user.is_active = active
    db.commit(); db.refresh(user)
    return user


@router.post("/users/{user_id}/reset-password", status_code=204)
def reset_password(
    user_id: int,
    new_password: str,
    db: Session = Depends(get_db),
    _: models.User = Depends(require_roles("admin")),
):
    if len(new_password) < 4:
        raise HTTPException(status_code=422, detail="Password is too short")
    user = db.get(models.User, user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    user.password_hash = hash_password(new_password)
    db.query(models.RefreshToken).filter(models.RefreshToken.user_id == user.id).delete()
    db.commit()
    return None


@router.post("/snapshots/rebuild")
def rebuild_snapshots(
    db: Session = Depends(get_db),
    _: models.User = Depends(require_roles("admin")),
):
    count = refresh_all_snapshots(db)
    return {"snapshots_refreshed": count}
