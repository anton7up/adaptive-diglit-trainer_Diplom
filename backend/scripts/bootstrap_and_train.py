#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json

from app.database import Base, SessionLocal, engine
from app.services.seed import seed_catalog, seed_synthetic_history
from app.services.training_orchestrator import train_from_database


def main() -> None:
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        catalog = seed_catalog(db)
        attempts = seed_synthetic_history(db)
        result = train_from_database(db, activate=True)
        print(json.dumps({
            "catalog": catalog,
            "synthetic_attempts_created": attempts,
            "model": result["model_name"],
            "version": result["version"],
            "dataset_rows": result["metadata"]["dataset_rows"],
            "candidates": [
                {"name": row["name"], "score": row["composite_score"], "validation": row["validation"]}
                for row in result["metadata"]["candidates"]
            ],
        }, ensure_ascii=False, indent=2))
    finally:
        db.close()


if __name__ == "__main__":
    main()
