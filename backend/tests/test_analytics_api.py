from __future__ import annotations

from app import models


def test_teacher_dashboard_requires_teacher(client, student_headers):
    assert client.get("/teacher/dashboard", headers=student_headers).status_code == 403


def test_teacher_dashboard_initial_state(client, teacher_headers):
    response = client.get("/teacher/dashboard", headers=teacher_headers)
    assert response.status_code == 200
    payload = response.json()
    assert payload["students_total"] == 1
    assert payload["attempts_total"] == 0
    assert len(payload["skills"]) == 6


def test_dashboard_after_student_attempt(client, student_headers, teacher_headers):
    task = client.post("/practice/next", json={"skill_id": None}, headers=student_headers).json()
    client.post("/practice/submit", json={"task_id": task["id"], "selected_option": "A", "response_time_ms": 2000}, headers=student_headers)
    dashboard = client.get("/teacher/dashboard", headers=teacher_headers).json()
    assert dashboard["attempts_total"] == 1
    assert dashboard["active_students_7d"] == 1
    assert len(dashboard["timeline"]) == 1


def test_student_progress_contains_six_skills(client, student_headers):
    response = client.get("/progress/me", headers=student_headers)
    assert response.status_code == 200
    assert len(response.json()["rows"]) == 6
