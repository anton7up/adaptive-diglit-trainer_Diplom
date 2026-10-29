from __future__ import annotations

from app import models


def test_skills_require_auth(client):
    assert client.get("/skills").status_code == 401


def test_skills_return_seeded_catalog(client, student_headers):
    response = client.get("/skills", headers=student_headers)
    assert response.status_code == 200
    assert len(response.json()) == 6
    assert {row["code"] for row in response.json()} == {"MEDIA", "SECURITY", "INTERNET", "DOCS", "TABLES", "LEGAL"}


def test_teacher_can_list_tasks(client, teacher_headers):
    response = client.get("/tasks", headers=teacher_headers)
    assert response.status_code == 200
    assert len(response.json()) == 120


def test_student_cannot_list_admin_task_payload(client, student_headers):
    assert client.get("/tasks", headers=student_headers).status_code == 403


def test_teacher_can_create_task(client, teacher_headers, db_session):
    skill = db_session.query(models.Skill).first()
    payload = {
        "skill_id": skill.id,
        "question": "Тестовый вопрос для проверки API?",
        "option_a": "A", "option_b": "B", "option_c": "C", "option_d": "D",
        "correct_option": "A",
        "explanation": "Пояснение",
        "difficulty": 0.44,
        "discrimination": 1.1,
        "cognitive_level": 2,
        "estimated_time_sec": 50,
        "tags": ["pytest"],
        "active": True,
    }
    response = client.post("/tasks", json=payload, headers=teacher_headers)
    assert response.status_code == 201
    assert response.json()["difficulty"] == 0.44


def test_teacher_can_patch_task(client, teacher_headers, db_session):
    task = db_session.query(models.Task).first()
    response = client.patch(f"/tasks/{task.id}", json={"difficulty": 0.77, "active": False}, headers=teacher_headers)
    assert response.status_code == 200
    assert response.json()["difficulty"] == 0.77
    assert response.json()["active"] is False
