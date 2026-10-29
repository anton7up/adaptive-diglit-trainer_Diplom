from __future__ import annotations

from datetime import datetime, timedelta

from app.timeutils import utcnow_naive
from app import models
from app.services.item_analysis import _point_biserial, analyze_item_bank, summarize_item_bank


def test_point_biserial_requires_variation():
    assert _point_biserial([1] * 10, [0.5] * 10) is None


def test_point_biserial_positive_for_discriminating_item():
    outcomes = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
    totals = [0.1, 0.2, 0.25, 0.3, 0.6, 0.7, 0.75, 0.8, 0.9, 0.95]
    value = _point_biserial(outcomes, totals)
    assert value is not None
    assert value > 0.5


def test_item_analysis_returns_whole_bank(db_session):
    rows = analyze_item_bank(db_session)
    assert len(rows) == 120
    assert all("configured_difficulty" in row for row in rows)
    assert all("distractor_counts" in row for row in rows)


def test_item_analysis_uses_attempts(db_session):
    user = db_session.query(models.User).filter_by(username="student_demo").one()
    task = db_session.query(models.Task).first()
    now = utcnow_naive()
    for index in range(12):
        correct = index < 9
        db_session.add(models.Attempt(
            user_id=user.id,
            task_id=task.id,
            selected_option=task.correct_option if correct else ("B" if task.correct_option != "B" else "C"),
            is_correct=correct,
            response_time_ms=5000 + index * 100,
            created_at=now + timedelta(seconds=index),
        ))
    db_session.commit()
    row = next(item for item in analyze_item_bank(db_session) if item["task_id"] == task.id)
    assert row["attempts"] == 12
    assert row["empirical_success_rate"] == 0.75
    assert row["median_response_time_ms"] == 5550.0


def test_item_summary_counts_observed(db_session):
    user = db_session.query(models.User).filter_by(username="student_demo").one()
    task = db_session.query(models.Task).first()
    db_session.add(models.Attempt(user_id=user.id, task_id=task.id, selected_option="A", is_correct=False, response_time_ms=1000))
    db_session.commit()
    summary = summarize_item_bank(analyze_item_bank(db_session))
    assert summary["items_total"] == 120
    assert summary["items_with_observations"] == 1


def test_teacher_item_endpoint(client, teacher_headers):
    response = client.get("/teacher/items/analysis", headers=teacher_headers)
    assert response.status_code == 200
    assert response.json()["summary"]["items_total"] == 120


def test_student_cannot_open_item_analysis(client, student_headers):
    assert client.get("/teacher/items/analysis", headers=student_headers).status_code == 403
