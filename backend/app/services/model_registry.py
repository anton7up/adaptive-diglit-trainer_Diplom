from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
from sqlalchemy.orm import Session

from .. import models
from ..config import get_settings


class ModelRegistry:
    def __init__(self, db: Session):
        self.db = db
        self.settings = get_settings()

    def _resolve_artifact_path(self, stored_path: str, *, metadata: bool = False) -> Path:
        path = Path(stored_path)
        if path.exists():
            return path
        fallback_dir = self.settings.model_dir
        candidate = fallback_dir / path.name
        if candidate.exists():
            return candidate
        suffix = ".json" if metadata else ".joblib"
        if path.suffix != suffix:
            candidate = fallback_dir / f"{path.stem}{suffix}"
            if candidate.exists():
                return candidate
        return path

    def register(self, result: dict[str, Any], activate: bool = True) -> models.MLModelVersion:
        if activate:
            self.db.query(models.MLModelVersion).update({models.MLModelVersion.is_active: False})
        metadata = result["metadata"]
        row = models.MLModelVersion(
            version=result["version"],
            model_name=result["model_name"],
            artifact_path=result["model_path"],
            metadata_path=result["metadata_path"],
            training_run_id=result["run_id"],
            is_active=activate,
            metrics=metadata["candidates"][0],
            feature_names=metadata["feature_names"],
            dataset_hash=metadata["dataset_hash"],
        )
        self.db.add(row)
        self.db.commit()
        self.db.refresh(row)
        return row

    def active_record(self) -> models.MLModelVersion | None:
        return (
            self.db.query(models.MLModelVersion)
            .filter(models.MLModelVersion.is_active.is_(True))
            .order_by(models.MLModelVersion.trained_at.desc())
            .first()
        )

    def activate(self, version: str) -> models.MLModelVersion:
        row = self.db.query(models.MLModelVersion).filter_by(version=version).first()
        if row is None:
            raise ValueError(f"unknown model version: {version}")
        self.db.query(models.MLModelVersion).update({models.MLModelVersion.is_active: False})
        row.is_active = True
        self.db.commit()
        return row

    def load_active(self):
        row = self.active_record()
        if row is None:
            return None, None
        path = self._resolve_artifact_path(row.artifact_path)
        if not path.exists():
            raise FileNotFoundError(f"active model artifact is missing: {path}")
        return row, joblib.load(path)

    def metadata(self, version: str) -> dict[str, Any]:
        row = self.db.query(models.MLModelVersion).filter_by(version=version).first()
        if row is None:
            raise ValueError(version)
        path = self._resolve_artifact_path(row.metadata_path, metadata=True)
        if not path.exists():
            raise FileNotFoundError(f"model metadata is missing: {path}")
        return json.loads(path.read_text(encoding="utf-8"))
