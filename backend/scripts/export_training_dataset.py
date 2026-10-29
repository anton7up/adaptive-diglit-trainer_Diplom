#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from sqlalchemy.orm import joinedload

from app import models
from app.database import SessionLocal
from app.services.feature_engineering import build_training_frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/reports/training_dataset.csv"))
    args = parser.parse_args()
    db = SessionLocal()
    try:
        attempts = db.query(models.Attempt).options(joinedload(models.Attempt.task).joinedload(models.Task.skill)).all()
        frame = build_training_frame(attempts)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(args.output, index=False)
        print(f"rows={len(frame)} output={args.output}")
    finally:
        db.close()


if __name__ == "__main__":
    main()
