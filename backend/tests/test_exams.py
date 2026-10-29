from __future__ import annotations

from app import models


def test_exams_are_listed(client, student_headers):
    response = client.get("/exams", headers=student_headers)
    assert response.status_code == 200
    assert len(response.json()) == 1


def test_student_can_start_exam(client, student_headers):
    exam_id = client.get("/exams", headers=student_headers).json()[0]["id"]
    response = client.post(f"/exams/{exam_id}/start", headers=student_headers)
    assert response.status_code == 200
    payload = response.json()
    assert payload["exam_id"] == exam_id
    assert len(payload["tasks"]) == 12
    assert "correct_option" not in payload["tasks"][0]


def test_start_reuses_unexpired_session(client, student_headers):
    exam_id = client.get("/exams", headers=student_headers).json()[0]["id"]
    first = client.post(f"/exams/{exam_id}/start", headers=student_headers).json()
    second = client.post(f"/exams/{exam_id}/start", headers=student_headers).json()
    assert first["session_id"] == second["session_id"]


def test_exam_answer_is_recorded(client, student_headers, db_session):
    exam_id = client.get("/exams", headers=student_headers).json()[0]["id"]
    session = client.post(f"/exams/{exam_id}/start", headers=student_headers).json()
    task = session["tasks"][0]
    response = client.post(f"/exams/sessions/{session['session_id']}/answers", json={
        "task_id": task["id"], "selected_option": "A", "response_time_ms": 5000,
    }, headers=student_headers)
    assert response.status_code == 204
    assert db_session.query(models.ExamAnswer).count() == 1


def test_duplicate_exam_answer_is_rejected(client, student_headers):
    exam_id = client.get("/exams", headers=student_headers).json()[0]["id"]
    session = client.post(f"/exams/{exam_id}/start", headers=student_headers).json()
    task_id = session["tasks"][0]["id"]
    body = {"task_id": task_id, "selected_option": "A", "response_time_ms": 1000}
    assert client.post(f"/exams/sessions/{session['session_id']}/answers", json=body, headers=student_headers).status_code == 204
    assert client.post(f"/exams/sessions/{session['session_id']}/answers", json=body, headers=student_headers).status_code == 409


def test_exam_submission_calculates_score(client, student_headers, db_session):
    exam_id = client.get("/exams", headers=student_headers).json()[0]["id"]
    session = client.post(f"/exams/{exam_id}/start", headers=student_headers).json()
    for task_payload in session["tasks"][:3]:
        task = db_session.get(models.Task, task_payload["id"])
        client.post(f"/exams/sessions/{session['session_id']}/answers", json={
            "task_id": task.id,
            "selected_option": task.correct_option,
            "response_time_ms": 1000,
        }, headers=student_headers)
    response = client.post(f"/exams/sessions/{session['session_id']}/submit", headers=student_headers)
    assert response.status_code == 200
    assert response.json()["correct_count"] == 3
    assert response.json()["total_count"] == 12
    assert response.json()["score"] == 0.25


def test_teacher_can_create_exam(client, teacher_headers, db_session):
    task_ids = [row.id for row in db_session.query(models.Task).limit(4).all()]
    response = client.post("/exams", json={
        "title": "Новый контроль",
        "description": "Тестовый экзамен",
        "duration_minutes": 20,
        "passing_score": 0.75,
        "task_ids": task_ids,
    }, headers=teacher_headers)
    assert response.status_code == 201
    assert response.json()["title"] == "Новый контроль"


def test_student_cannot_create_exam(client, student_headers, db_session):
    task_ids = [row.id for row in db_session.query(models.Task).limit(2).all()]
    response = client.post("/exams", json={"title": "Forbidden", "task_ids": task_ids}, headers=student_headers)
    assert response.status_code == 403
