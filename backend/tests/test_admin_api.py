from __future__ import annotations

from app import models
from app.security import verify_password


def test_non_admin_cannot_list_users(client, teacher_headers):
    assert client.get("/admin/users", headers=teacher_headers).status_code == 403


def test_admin_can_list_users(client, admin_headers):
    response = client.get("/admin/users", headers=admin_headers)
    assert response.status_code == 200
    assert len(response.json()) == 3


def test_admin_can_promote_student(client, admin_headers, db_session):
    student = db_session.query(models.User).filter_by(username="student_demo").one()
    response = client.patch(f"/admin/users/{student.id}/role", params={"role": "teacher"}, headers=admin_headers)
    assert response.status_code == 200
    assert response.json()["role"] == "teacher"


def test_admin_cannot_remove_own_admin_role(client, admin_headers, db_session):
    admin = db_session.query(models.User).filter_by(username="admin_demo").one()
    response = client.patch(f"/admin/users/{admin.id}/role", params={"role": "student"}, headers=admin_headers)
    assert response.status_code == 409


def test_admin_can_deactivate_other_user(client, admin_headers, db_session):
    student = db_session.query(models.User).filter_by(username="student_demo").one()
    response = client.patch(f"/admin/users/{student.id}/active", params={"active": "false"}, headers=admin_headers)
    assert response.status_code == 200
    assert response.json()["is_active"] is False


def test_admin_password_reset_rehashes_password(client, admin_headers, db_session):
    student = db_session.query(models.User).filter_by(username="student_demo").one()
    old_hash = student.password_hash
    response = client.post(f"/admin/users/{student.id}/reset-password", params={"new_password": "newpass123"}, headers=admin_headers)
    assert response.status_code == 204
    db_session.refresh(student)
    assert student.password_hash != old_hash
    assert verify_password("newpass123", student.password_hash)
