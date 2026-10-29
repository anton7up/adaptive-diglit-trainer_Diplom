#!/usr/bin/env python3
"""Import users, skills, tasks and attempts from the previous SQLite schema.

The importer is intentionally explicit: the legacy database is read-only, passwords are
not silently downgraded, and collisions are reported instead of overwritten.
"""
from __future__ import annotations

import argparse
import sqlite3
from datetime import datetime
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import models
from app.database import SessionLocal
from app.security import hash_password


def rows(connection: sqlite3.Connection, table: str):
    connection.row_factory = sqlite3.Row
    return connection.execute(f"SELECT * FROM {table}").fetchall()


def import_database(path: Path) -> dict[str, int]:
    source = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    db = SessionLocal()
    counts = {"users": 0, "skills": 0, "tasks": 0, "attempts": 0}
    try:
        skill_map: dict[int, int] = {}
        for item in rows(source, "skills"):
            existing = db.query(models.Skill).filter_by(code=item["code"]).first()
            if existing is None:
                existing = models.Skill(code=item["code"], name=item["name"], description=item["description"])
                db.add(existing); db.flush(); counts["skills"] += 1
            skill_map[item["id"]] = existing.id

        task_map: dict[int, int] = {}
        for item in rows(source, "tasks"):
            existing = db.query(models.Task).filter_by(question=item["question"]).first()
            if existing is None:
                difficulty_old = float(item["difficulty"] or 0.0)
                difficulty = max(0.05, min(0.95, 0.5 + difficulty_old * 0.22))
                existing = models.Task(
                    skill_id=skill_map[item["skill_id"]], question=item["question"], option_a=item["option_a"],
                    option_b=item["option_b"], option_c=item["option_c"], option_d=item["option_d"],
                    correct_option=item["correct_option"], explanation=item["explanation"],
                    source_reference=item["source_reference"], difficulty=difficulty, active=bool(item["active"]),
                    tags=["legacy-import"],
                )
                db.add(existing); db.flush(); counts["tasks"] += 1
            task_map[item["id"]] = existing.id

        user_map: dict[int, int] = {}
        for item in rows(source, "users"):
            username = item["username"]
            existing = db.query(models.User).filter_by(username=username).first()
            if existing is None:
                existing = models.User(
                    username=username,
                    password_hash=hash_password("change-me-after-import"),
                    role=item["role"],
                    full_name=f"{username} (legacy import)",
                )
                db.add(existing); db.flush(); counts["users"] += 1
            user_map[item["id"]] = existing.id

        for item in rows(source, "attempts"):
            if item["user_id"] not in user_map or item["task_id"] not in task_map:
                continue
            exists = db.query(models.Attempt).filter_by(
                user_id=user_map[item["user_id"]], task_id=task_map[item["task_id"]], created_at=item["created_at"]
            ).first()
            if exists:
                continue
            db.add(models.Attempt(
                user_id=user_map[item["user_id"]], task_id=task_map[item["task_id"]],
                selected_option=item["selected_option"], is_correct=bool(item["is_correct"]),
                predicted_probability=item["predicted_probability"], prediction_model="legacy-irt-elo",
                response_time_ms=item["response_time_ms"] or 0,
                created_at=datetime.fromisoformat(item["created_at"]) if isinstance(item["created_at"], str) else item["created_at"],
                client_context={"source": "legacy_db_import"},
            )); counts["attempts"] += 1
        db.commit()
        return counts
    finally:
        db.close(); source.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("legacy_db", type=Path)
    args = parser.parse_args()
    print(import_database(args.legacy_db))


if __name__ == "__main__":
    main()
