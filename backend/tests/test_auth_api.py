from __future__ import annotations


def test_login_returns_jwt_pair(client):
    response = client.post("/auth/login", json={"username": "student_demo", "password": "1234"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["token_type"] == "bearer"
    assert payload["access_token"]
    assert payload["refresh_token"]
    assert payload["user"]["role"] == "student"


def test_login_rejects_wrong_password(client):
    response = client.post("/auth/login", json={"username": "student_demo", "password": "bad"})
    assert response.status_code == 401


def test_registration_creates_student(client):
    response = client.post("/auth/register", json={
        "username": "new_student",
        "password": "abcd1234",
        "full_name": "Новый Студент",
        "group_name": "ИТ-101",
    })
    assert response.status_code == 201
    assert response.json()["user"]["role"] == "student"
    assert response.json()["user"]["group_name"] == "ИТ-101"


def test_duplicate_registration_is_conflict(client):
    body = {"username": "duplicate_user", "password": "abcd"}
    assert client.post("/auth/register", json=body).status_code == 201
    assert client.post("/auth/register", json=body).status_code == 409


def test_me_requires_authentication(client):
    assert client.get("/auth/me").status_code == 401


def test_me_returns_current_user(client, student_headers):
    response = client.get("/auth/me", headers=student_headers)
    assert response.status_code == 200
    assert response.json()["username"] == "student_demo"


def test_refresh_rotates_refresh_token(client):
    login = client.post("/auth/login", json={"username": "student_demo", "password": "1234"}).json()
    first_refresh = login["refresh_token"]
    response = client.post("/auth/refresh", json={"refresh_token": first_refresh})
    assert response.status_code == 200
    second_refresh = response.json()["refresh_token"]
    assert second_refresh != first_refresh
    replay = client.post("/auth/refresh", json={"refresh_token": first_refresh})
    assert replay.status_code == 401


def test_logout_revokes_refresh_token(client):
    login = client.post("/auth/login", json={"username": "student_demo", "password": "1234"}).json()
    refresh = login["refresh_token"]
    assert client.post("/auth/logout", json={"refresh_token": refresh}).status_code == 204
    assert client.post("/auth/refresh", json={"refresh_token": refresh}).status_code == 401
