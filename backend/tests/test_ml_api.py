from __future__ import annotations


def test_ml_status_requires_teacher(client, student_headers):
    assert client.get("/ml/status", headers=student_headers).status_code == 403


def test_ml_status_without_model(client, teacher_headers):
    response = client.get("/ml/status", headers=teacher_headers)
    assert response.status_code == 200
    assert response.json()["active"] is False


def test_ml_quality_with_empty_history_returns_report(client, teacher_headers):
    response = client.get("/ml/quality", headers=teacher_headers)
    assert response.status_code == 200
    payload = response.json()
    assert payload["passed"] is False
    assert payload["rows"] == 0


def test_admin_can_bootstrap(client, admin_headers):
    response = client.post("/ml/bootstrap", json={"users": 5, "attempts_per_user": 10}, headers=admin_headers)
    assert response.status_code == 200
    assert response.json()["attempts_created"] == 50


def test_teacher_cannot_bootstrap(client, teacher_headers):
    response = client.post("/ml/bootstrap", json={"users": 5, "attempts_per_user": 10}, headers=teacher_headers)
    assert response.status_code == 403


def test_retraining_policy_without_model_requests_training(client, teacher_headers):
    response = client.get("/ml/retraining-policy", headers=teacher_headers)
    assert response.status_code == 200
    assert response.json()["should_retrain"] is True
    assert "no_active_model" in response.json()["reasons"]


def test_research_endpoint_without_active_model_is_404(client, teacher_headers):
    assert client.get("/research/model-card", headers=teacher_headers).status_code == 404
