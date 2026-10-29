#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json
from sqlalchemy.orm import joinedload

from app import models
from app.database import SessionLocal
from app.services.ablation import run_ablation
from app.services.feature_engineering import build_training_frame


def main() -> None:
    db = SessionLocal()
    try:
        attempts = db.query(models.Attempt).options(joinedload(models.Attempt.task).joinedload(models.Task.skill)).all()
        frame = build_training_frame(attempts)
        print(json.dumps(run_ablation(frame), ensure_ascii=False, indent=2))
    finally:
        db.close()


if __name__ == "__main__":
    main()
