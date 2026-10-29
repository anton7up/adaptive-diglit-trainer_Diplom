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
from app.services.feature_engineering import build_training_frame
from app.services.model_evaluation import build_evaluation_report, save_evaluation_report
from app.services.model_registry import ModelRegistry


def main() -> None:
    db = SessionLocal()
    try:
        record, estimator = ModelRegistry(db).load_active()
        if record is None:
            raise SystemExit("No active model")
        attempts = (
            db.query(models.Attempt)
            .options(joinedload(models.Attempt.task).joinedload(models.Task.skill))
            .order_by(models.Attempt.created_at.asc(), models.Attempt.id.asc())
            .all()
        )
        frame = build_training_frame(attempts)
        report = build_evaluation_report(estimator, frame, record.version)
        path = save_evaluation_report(report)
        print(json.dumps({"report": str(path), "metrics": report["metrics"], "ece": report["expected_calibration_error"]}, indent=2))
    finally:
        db.close()


if __name__ == "__main__":
    main()
