from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

Role = Literal["student", "teacher", "admin"]


class ORMModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)


class RegisterRequest(BaseModel):
    username: str = Field(min_length=3, max_length=64, pattern=r"^[A-Za-z0-9_.-]+$")
    password: str = Field(min_length=4, max_length=128)
    full_name: str | None = Field(default=None, max_length=160)
    group_name: str | None = Field(default=None, max_length=80)


class LoginRequest(BaseModel):
    username: str
    password: str


class RefreshRequest(BaseModel):
    refresh_token: str


class LogoutRequest(BaseModel):
    refresh_token: str


class UserResponse(ORMModel):
    id: int
    username: str
    role: str
    full_name: str | None = None
    group_name: str | None = None
    is_active: bool
    created_at: datetime


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in_seconds: int
    user: UserResponse


class SkillResponse(ORMModel):
    id: int
    code: str
    name: str
    description: str | None = None
    sort_order: int
    active: bool


class TaskBase(BaseModel):
    skill_id: int
    question: str = Field(min_length=5)
    option_a: str
    option_b: str
    option_c: str
    option_d: str
    correct_option: str = Field(pattern=r"^[ABCD]$")
    explanation: str
    source_reference: str | None = None
    difficulty: float = Field(ge=0.0, le=1.0)
    discrimination: float = Field(default=1.0, ge=0.1, le=3.0)
    cognitive_level: int = Field(default=1, ge=1, le=3)
    estimated_time_sec: int = Field(default=45, ge=5, le=600)
    tags: list[str] = Field(default_factory=list)
    active: bool = True


class TaskCreate(TaskBase):
    pass


class TaskUpdate(BaseModel):
    question: str | None = None
    option_a: str | None = None
    option_b: str | None = None
    option_c: str | None = None
    option_d: str | None = None
    correct_option: str | None = Field(default=None, pattern=r"^[ABCD]$")
    explanation: str | None = None
    source_reference: str | None = None
    difficulty: float | None = Field(default=None, ge=0.0, le=1.0)
    discrimination: float | None = Field(default=None, ge=0.1, le=3.0)
    cognitive_level: int | None = Field(default=None, ge=1, le=3)
    estimated_time_sec: int | None = Field(default=None, ge=5, le=600)
    tags: list[str] | None = None
    active: bool | None = None


class TaskAdminResponse(ORMModel):
    id: int
    skill_id: int
    question: str
    option_a: str
    option_b: str
    option_c: str
    option_d: str
    correct_option: str
    explanation: str
    source_reference: str | None
    difficulty: float
    discrimination: float
    cognitive_level: int
    estimated_time_sec: int
    tags: list[str]
    active: bool


class AdaptiveTaskResponse(BaseModel):
    id: int
    skill_id: int
    skill_name: str
    question: str
    options: dict[str, str]
    difficulty: float
    cognitive_level: int
    estimated_time_sec: int
    predicted_probability: float
    model_version: str
    selection_score: float
    selection_components: dict[str, float]


class NextTaskRequest(BaseModel):
    skill_id: int | None = None


class SubmitAnswerRequest(BaseModel):
    task_id: int
    selected_option: str = Field(pattern=r"^[ABCD]$")
    response_time_ms: int = Field(default=0, ge=0, le=3_600_000)
    hint_used: bool = False
    session_id: str | None = Field(default=None, max_length=64)


class SubmitAnswerResponse(BaseModel):
    attempt_id: int
    is_correct: bool
    correct_option: str
    explanation: str
    predicted_probability: float | None
    model_version: str | None


class SkillProgressRow(BaseModel):
    skill_id: int
    skill_code: str
    skill_name: str
    attempts_count: int
    correct_count: int
    accuracy: float
    rolling_accuracy_5: float
    rolling_accuracy_10: float
    mean_response_time_ms: float
    mastery_estimate: float


class ProgressResponse(BaseModel):
    user_id: int
    username: str
    attempts_total: int
    correct_total: int
    overall_accuracy: float
    rows: list[SkillProgressRow]


class TimelinePoint(BaseModel):
    date: str
    attempts: int
    correct: int
    accuracy: float


class TeacherStudentRow(BaseModel):
    user_id: int
    username: str
    full_name: str | None
    attempts: int
    accuracy: float
    avg_response_time_ms: float
    last_activity_at: datetime | None


class TeacherSkillRow(BaseModel):
    skill_id: int
    code: str
    name: str
    attempts: int
    accuracy: float
    avg_predicted_probability: float | None
    calibration_gap: float | None


class TeacherDashboardResponse(BaseModel):
    students_total: int
    attempts_total: int
    active_students_7d: int
    overall_accuracy: float
    students: list[TeacherStudentRow]
    skills: list[TeacherSkillRow]
    timeline: list[TimelinePoint]


class ExamSummary(ORMModel):
    id: int
    title: str
    description: str | None
    duration_minutes: int
    passing_score: float
    active: bool
    created_at: datetime


class ExamCreate(BaseModel):
    title: str = Field(min_length=3, max_length=180)
    description: str | None = None
    duration_minutes: int = Field(default=30, ge=1, le=240)
    passing_score: float = Field(default=0.6, ge=0.0, le=1.0)
    task_ids: list[int] = Field(min_length=1)


class ExamStartResponse(BaseModel):
    session_id: int
    exam_id: int
    title: str
    duration_minutes: int
    started_at: datetime
    tasks: list[dict[str, Any]]


class ExamAnswerRequest(BaseModel):
    task_id: int
    selected_option: str = Field(pattern=r"^[ABCD]$")
    response_time_ms: int = Field(default=0, ge=0, le=3_600_000)


class ExamSubmitResponse(BaseModel):
    session_id: int
    score: float
    passed: bool
    correct_count: int
    total_count: int
    submitted_at: datetime


class ModelCandidateMetrics(BaseModel):
    name: str
    composite_score: float
    validation: dict[str, float]
    cross_validation_mean: dict[str, float]


class ModelStatusResponse(BaseModel):
    active: bool
    version: str | None = None
    model_name: str | None = None
    trained_at: datetime | None = None
    dataset_hash: str | None = None
    metrics: dict[str, Any] | None = None
    feature_names: list[str] = Field(default_factory=list)


class TrainingResponse(BaseModel):
    run_id: str
    version: str
    model_name: str
    dataset_rows: int
    positive_rate: float
    candidates: list[ModelCandidateMetrics]


class PredictionAuditResponse(ORMModel):
    id: int
    user_id: int
    task_id: int
    model_version: str
    probability: float
    selection_score: float | None
    selection_components: dict[str, float]
    created_at: datetime


class SeedSyntheticRequest(BaseModel):
    users: int = Field(default=36, ge=5, le=200)
    attempts_per_user: int = Field(default=55, ge=10, le=300)


class SeedSyntheticResponse(BaseModel):
    attempts_created: int
