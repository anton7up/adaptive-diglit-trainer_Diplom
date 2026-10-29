from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    JSON,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import relationship

from .database import Base
from .timeutils import utcnow_naive


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    username = Column(String(64), unique=True, index=True, nullable=False)
    password_hash = Column(String(256), nullable=False)
    role = Column(String(16), nullable=False, default="student", index=True)
    full_name = Column(String(160), nullable=True)
    group_name = Column(String(80), nullable=True)
    is_active = Column(Boolean, nullable=False, default=True)
    created_at = Column(DateTime, nullable=False, default=utcnow_naive)
    last_login_at = Column(DateTime, nullable=True)

    attempts = relationship("Attempt", back_populates="user", cascade="all, delete-orphan")
    exam_sessions = relationship("ExamSession", back_populates="user", cascade="all, delete-orphan")
    tokens = relationship("RefreshToken", back_populates="user", cascade="all, delete-orphan")


class RefreshToken(Base):
    __tablename__ = "refresh_tokens"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    jti = Column(String(64), nullable=False, unique=True, index=True)
    expires_at = Column(DateTime, nullable=False)
    revoked_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, nullable=False, default=utcnow_naive)

    user = relationship("User", back_populates="tokens")


class Skill(Base):
    __tablename__ = "skills"

    id = Column(Integer, primary_key=True)
    code = Column(String(32), unique=True, nullable=False, index=True)
    name = Column(String(128), nullable=False)
    description = Column(Text, nullable=True)
    sort_order = Column(Integer, nullable=False, default=0)
    active = Column(Boolean, nullable=False, default=True)

    tasks = relationship("Task", back_populates="skill", cascade="all, delete-orphan")


class Task(Base):
    __tablename__ = "tasks"

    id = Column(Integer, primary_key=True)
    skill_id = Column(Integer, ForeignKey("skills.id"), nullable=False, index=True)
    question = Column(Text, nullable=False)
    option_a = Column(Text, nullable=False)
    option_b = Column(Text, nullable=False)
    option_c = Column(Text, nullable=False)
    option_d = Column(Text, nullable=False)
    correct_option = Column(String(1), nullable=False)
    explanation = Column(Text, nullable=False)
    source_reference = Column(String(256), nullable=True)
    difficulty = Column(Float, nullable=False, default=0.5)
    discrimination = Column(Float, nullable=False, default=1.0)
    cognitive_level = Column(Integer, nullable=False, default=1)
    estimated_time_sec = Column(Integer, nullable=False, default=45)
    tags = Column(JSON, nullable=False, default=list)
    active = Column(Boolean, nullable=False, default=True, index=True)
    created_at = Column(DateTime, nullable=False, default=utcnow_naive)

    skill = relationship("Skill", back_populates="tasks")
    attempts = relationship("Attempt", back_populates="task", cascade="all, delete-orphan")
    exam_links = relationship("ExamTask", back_populates="task", cascade="all, delete-orphan")


class Attempt(Base):
    __tablename__ = "attempts"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    task_id = Column(Integer, ForeignKey("tasks.id"), nullable=False, index=True)
    session_id = Column(String(64), nullable=True, index=True)
    selected_option = Column(String(1), nullable=False)
    is_correct = Column(Boolean, nullable=False, index=True)
    predicted_probability = Column(Float, nullable=True)
    prediction_model = Column(String(128), nullable=True)
    response_time_ms = Column(Integer, nullable=False, default=0)
    hint_used = Column(Boolean, nullable=False, default=False)
    skipped_before = Column(Integer, nullable=False, default=0)
    client_context = Column(JSON, nullable=False, default=dict)
    created_at = Column(DateTime, nullable=False, default=utcnow_naive, index=True)

    user = relationship("User", back_populates="attempts")
    task = relationship("Task", back_populates="attempts")


class UserSkillSnapshot(Base):
    __tablename__ = "user_skill_snapshots"
    __table_args__ = (UniqueConstraint("user_id", "skill_id", name="uq_user_skill_snapshot"),)

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    skill_id = Column(Integer, ForeignKey("skills.id"), nullable=False, index=True)
    attempts_count = Column(Integer, nullable=False, default=0)
    correct_count = Column(Integer, nullable=False, default=0)
    rolling_accuracy_5 = Column(Float, nullable=False, default=0.5)
    rolling_accuracy_10 = Column(Float, nullable=False, default=0.5)
    mean_response_time_ms = Column(Float, nullable=False, default=0.0)
    recent_response_time_ms = Column(Float, nullable=False, default=0.0)
    streak_correct = Column(Integer, nullable=False, default=0)
    streak_incorrect = Column(Integer, nullable=False, default=0)
    days_since_last_attempt = Column(Float, nullable=False, default=30.0)
    mastery_estimate = Column(Float, nullable=False, default=0.5)
    updated_at = Column(DateTime, nullable=False, default=utcnow_naive, onupdate=utcnow_naive)


class Exam(Base):
    __tablename__ = "exams"

    id = Column(Integer, primary_key=True)
    title = Column(String(180), nullable=False)
    description = Column(Text, nullable=True)
    duration_minutes = Column(Integer, nullable=False, default=30)
    passing_score = Column(Float, nullable=False, default=0.6)
    active = Column(Boolean, nullable=False, default=True)
    created_by = Column(Integer, ForeignKey("users.id"), nullable=True)
    created_at = Column(DateTime, nullable=False, default=utcnow_naive)

    tasks = relationship("ExamTask", back_populates="exam", cascade="all, delete-orphan")
    sessions = relationship("ExamSession", back_populates="exam", cascade="all, delete-orphan")


class ExamTask(Base):
    __tablename__ = "exam_tasks"
    __table_args__ = (UniqueConstraint("exam_id", "task_id", name="uq_exam_task"),)

    id = Column(Integer, primary_key=True)
    exam_id = Column(Integer, ForeignKey("exams.id"), nullable=False, index=True)
    task_id = Column(Integer, ForeignKey("tasks.id"), nullable=False, index=True)
    position = Column(Integer, nullable=False)
    points = Column(Float, nullable=False, default=1.0)

    exam = relationship("Exam", back_populates="tasks")
    task = relationship("Task", back_populates="exam_links")


class ExamSession(Base):
    __tablename__ = "exam_sessions"

    id = Column(Integer, primary_key=True)
    exam_id = Column(Integer, ForeignKey("exams.id"), nullable=False, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    started_at = Column(DateTime, nullable=False, default=utcnow_naive)
    submitted_at = Column(DateTime, nullable=True)
    score = Column(Float, nullable=True)
    passed = Column(Boolean, nullable=True)
    status = Column(String(24), nullable=False, default="in_progress")

    exam = relationship("Exam", back_populates="sessions")
    user = relationship("User", back_populates="exam_sessions")
    answers = relationship("ExamAnswer", back_populates="session", cascade="all, delete-orphan")


class ExamAnswer(Base):
    __tablename__ = "exam_answers"
    __table_args__ = (UniqueConstraint("session_id", "task_id", name="uq_exam_session_task"),)

    id = Column(Integer, primary_key=True)
    session_id = Column(Integer, ForeignKey("exam_sessions.id"), nullable=False, index=True)
    task_id = Column(Integer, ForeignKey("tasks.id"), nullable=False, index=True)
    selected_option = Column(String(1), nullable=False)
    is_correct = Column(Boolean, nullable=False)
    points_awarded = Column(Float, nullable=False, default=0.0)
    response_time_ms = Column(Integer, nullable=False, default=0)
    answered_at = Column(DateTime, nullable=False, default=utcnow_naive)

    session = relationship("ExamSession", back_populates="answers")


class MLTrainingRun(Base):
    __tablename__ = "ml_training_runs"

    id = Column(Integer, primary_key=True)
    run_id = Column(String(64), nullable=False, unique=True, index=True)
    started_at = Column(DateTime, nullable=False, default=utcnow_naive)
    finished_at = Column(DateTime, nullable=True)
    status = Column(String(24), nullable=False, default="running")
    dataset_rows = Column(Integer, nullable=False, default=0)
    dataset_positive_rate = Column(Float, nullable=True)
    feature_count = Column(Integer, nullable=False, default=0)
    split_strategy = Column(String(64), nullable=False, default="group_temporal")
    best_model_name = Column(String(128), nullable=True)
    best_model_version = Column(String(96), nullable=True)
    selection_metric = Column(String(64), nullable=False, default="composite")
    metrics = Column(JSON, nullable=False, default=dict)
    parameters = Column(JSON, nullable=False, default=dict)
    notes = Column(Text, nullable=True)


class MLModelVersion(Base):
    __tablename__ = "ml_model_versions"

    id = Column(Integer, primary_key=True)
    version = Column(String(96), nullable=False, unique=True, index=True)
    model_name = Column(String(128), nullable=False)
    artifact_path = Column(String(512), nullable=False)
    metadata_path = Column(String(512), nullable=False)
    trained_at = Column(DateTime, nullable=False, default=utcnow_naive)
    training_run_id = Column(String(64), nullable=False, index=True)
    is_active = Column(Boolean, nullable=False, default=False, index=True)
    metrics = Column(JSON, nullable=False, default=dict)
    feature_names = Column(JSON, nullable=False, default=list)
    dataset_hash = Column(String(128), nullable=True)


class PredictionAudit(Base):
    __tablename__ = "prediction_audit"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    task_id = Column(Integer, ForeignKey("tasks.id"), nullable=False, index=True)
    model_version = Column(String(96), nullable=False, index=True)
    probability = Column(Float, nullable=False)
    feature_vector = Column(JSON, nullable=False, default=dict)
    selection_score = Column(Float, nullable=True)
    selection_components = Column(JSON, nullable=False, default=dict)
    created_at = Column(DateTime, nullable=False, default=utcnow_naive, index=True)
