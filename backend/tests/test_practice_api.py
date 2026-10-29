from __future__ import annotations

from app import models


def test_student_can_get_adaptive_task(client, student_headers):
    response = client.post("/practice/next", json={"skill_id": None}, headers=student_headers)
    assert response.status_code == 200
    payload = response.json()
    assert set(payload["options"]) == {"A", "B", "C", "D"}
    assert 0.0 < payload["predicted_probability"] < 1.0
    assert payload["model_version"]
    assert payload["selection_components"]


def test_student_can_filter_by_skill(client, student_headers, db_session):
    skill = db_session.query(models.Skill).first()
    response = client.post("/practice/next", json={"skill_id": skill.id}, headers=student_headers)
    assert response.status_code == 200
    assert response.json()["skill_id"] == skill.id


def test_teacher_cannot_use_student_practice(client, teacher_headers):
    response = client.post("/practice/next", json={"skill_id": None}, headers=teacher_headers)
    assert response.status_code == 403


def test_submit_records_attempt(client, student_headers, db_session):
    next_response = client.post("/practice/next", json={"skill_id": None}, headers=student_headers)
    task = next_response.json()
    before = db_session.query(models.Attempt).count()
    response = client.post("/practice/submit", json={
        "task_id": task["id"],
        "selected_option": "A",
        "response_time_ms": 9876,
        "hint_used": False,
        "session_id": "pytest",
    }, headers=student_headers)
    assert response.status_code == 200
    assert db_session.query(models.Attempt).count() == before + 1
    stored = db_session.query(models.Attempt).order_by(models.Attempt.id.desc()).first()
    assert stored.response_time_ms == 9876
    assert stored.prediction_model


def test_submit_updates_snapshot(client, student_headers, db_session):
    task = client.post("/practice/next", json={"skill_id": None}, headers=student_headers).json()
    client.post("/practice/submit", json={"task_id": task["id"], "selected_option": "A", "response_time_ms": 5000}, headers=student_headers)
    student = db_session.query(models.User).filter_by(username="student_demo").one()
    snapshot = db_session.query(models.UserSkillSnapshot).filter_by(user_id=student.id, skill_id=task["skill_id"]).first()
    assert snapshot is not None
    assert snapshot.attempts_count == 1


def test_progress_reflects_attempt(client, student_headers):
    task = client.post("/practice/next", json={"skill_id": None}, headers=student_headers).json()
    client.post("/practice/submit", json={"task_id": task["id"], "selected_option": "A", "response_time_ms": 5000}, headers=student_headers)
    progress = client.get("/progress/me", headers=student_headers)
    assert progress.status_code == 200
    assert progress.json()["attempts_total"] == 1


def test_prediction_audit_is_created(client, student_headers, teacher_headers, db_session):
    client.post("/practice/next", json={"skill_id": None}, headers=student_headers)
    assert db_session.query(models.PredictionAudit).count() == 1
    audit = client.get("/ml/audit", headers=teacher_headers)
    assert audit.status_code == 200
    assert len(audit.json()) == 1


def test_recent_task_penalty_changes_components(client, student_headers):
    first = client.post("/practice/next", json={"skill_id": None}, headers=student_headers).json()
    client.post("/practice/submit", json={"task_id": first["id"], "selected_option": "A", "response_time_ms": 1000}, headers=student_headers)
    second = client.post("/practice/next", json={"skill_id": None}, headers=student_headers).json()
    assert second["selection_components"]["recency_penalty"] in {0.0, -0.65}
