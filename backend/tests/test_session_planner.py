from __future__ import annotations

from app import models
from app.services.session_planner import build_practice_plan


def test_plan_has_requested_length(db_session):
    user = db_session.query(models.User).filter_by(username="student_demo").one()
    plan = build_practice_plan(db_session, user.id, length=8)
    assert plan["summary"]["length"] == 8
    assert len(plan["tasks"]) == 8
    assert len({row["task_id"] for row in plan["tasks"]}) == 8


def test_plan_covers_multiple_skills_in_mixed_mode(db_session):
    user = db_session.query(models.User).filter_by(username="student_demo").one()
    plan = build_practice_plan(db_session, user.id, length=12)
    assert plan["summary"]["skills_covered"] >= 4


def test_plan_can_be_restricted_to_skill(db_session):
    user = db_session.query(models.User).filter_by(username="student_demo").one()
    skill = db_session.query(models.Skill).first()
    plan = build_practice_plan(db_session, user.id, length=5, skill_id=skill.id)
    assert plan["summary"]["length"] == 5
    assert {row["skill_id"] for row in plan["tasks"]} == {skill.id}


def test_plan_probability_is_valid(db_session):
    user = db_session.query(models.User).filter_by(username="student_demo").one()
    plan = build_practice_plan(db_session, user.id, length=5)
    assert all(0.0 < row["probability"] < 1.0 for row in plan["tasks"])
    assert 0.0 < plan["summary"]["mean_predicted_success"] < 1.0


def test_plan_endpoint_requires_student(client, teacher_headers):
    assert client.get("/practice/plan", headers=teacher_headers).status_code == 403


def test_plan_endpoint_returns_batch(client, student_headers):
    response = client.get("/practice/plan", params={"length": 6}, headers=student_headers)
    assert response.status_code == 200
    assert response.json()["summary"]["length"] == 6
