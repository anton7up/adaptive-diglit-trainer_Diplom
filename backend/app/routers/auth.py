from __future__ import annotations

from datetime import datetime

import jwt
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from .. import models, schemas
from ..timeutils import utcnow_naive
from ..config import get_settings
from ..database import get_db
from ..dependencies import get_current_user
from ..security import create_access_token, create_refresh_token, decode_token, hash_password, verify_password

router = APIRouter(prefix="/auth", tags=["auth"])


def _issue_tokens(db: Session, user: models.User) -> schemas.TokenResponse:
    access = create_access_token(user.id, user.role, user.username)
    refresh, jti, expires_at = create_refresh_token(user.id, user.role, user.username)
    db.add(models.RefreshToken(user_id=user.id, jti=jti, expires_at=expires_at))
    db.commit()
    return schemas.TokenResponse(
        access_token=access,
        refresh_token=refresh,
        expires_in_seconds=get_settings().access_token_minutes * 60,
        user=schemas.UserResponse.model_validate(user),
    )


@router.post("/register", response_model=schemas.TokenResponse, status_code=201)
def register(payload: schemas.RegisterRequest, db: Session = Depends(get_db)):
    username = payload.username.strip().lower()
    if db.query(models.User).filter(models.User.username == username).first():
        raise HTTPException(status_code=409, detail="Username already exists")
    user = models.User(
        username=username,
        password_hash=hash_password(payload.password),
        role="student",
        full_name=payload.full_name,
        group_name=payload.group_name,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return _issue_tokens(db, user)


@router.post("/login", response_model=schemas.TokenResponse)
def login(payload: schemas.LoginRequest, db: Session = Depends(get_db)):
    user = db.query(models.User).filter(models.User.username == payload.username.strip().lower()).first()
    if user is None or not verify_password(payload.password, user.password_hash):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid username or password")
    if not user.is_active:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="User is inactive")
    user.last_login_at = utcnow_naive()
    db.commit()
    return _issue_tokens(db, user)


@router.post("/refresh", response_model=schemas.TokenResponse)
def refresh(payload: schemas.RefreshRequest, db: Session = Depends(get_db)):
    try:
        decoded = decode_token(payload.refresh_token, "refresh")
        user_id = int(decoded["sub"])
        jti = str(decoded["jti"])
    except (jwt.PyJWTError, KeyError, ValueError):
        raise HTTPException(status_code=401, detail="Invalid refresh token")
    token_row = db.query(models.RefreshToken).filter_by(jti=jti, user_id=user_id).first()
    if token_row is None or token_row.revoked_at is not None or token_row.expires_at <= utcnow_naive():
        raise HTTPException(status_code=401, detail="Refresh token is expired or revoked")
    token_row.revoked_at = utcnow_naive()
    user = db.get(models.User, user_id)
    if user is None or not user.is_active:
        raise HTTPException(status_code=401, detail="User unavailable")
    db.commit()
    return _issue_tokens(db, user)


@router.post("/logout", status_code=204)
def logout(payload: schemas.LogoutRequest, db: Session = Depends(get_db)):
    try:
        decoded = decode_token(payload.refresh_token, "refresh")
        row = db.query(models.RefreshToken).filter_by(jti=str(decoded["jti"])).first()
        if row is not None and row.revoked_at is None:
            row.revoked_at = utcnow_naive()
            db.commit()
    except jwt.PyJWTError:
        pass
    return None


@router.get("/me", response_model=schemas.UserResponse)
def me(user: models.User = Depends(get_current_user)):
    return user
