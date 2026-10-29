from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    app_name: str
    api_version: str
    database_url: str
    jwt_secret: str
    jwt_algorithm: str
    access_token_minutes: int
    refresh_token_days: int
    model_dir: Path
    report_dir: Path
    target_success_probability: float
    minimum_training_rows: int
    random_seed: int
    cors_origins: tuple[str, ...]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    base_dir = Path(__file__).resolve().parents[1]
    data_dir = base_dir / "data"
    model_dir = Path(os.getenv("ADAPTIVE_MODEL_DIR", str(data_dir / "model_artifacts"))).resolve()
    report_dir = Path(os.getenv("ADAPTIVE_REPORT_DIR", str(data_dir / "reports"))).resolve()
    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    origins_raw = os.getenv("ADAPTIVE_CORS_ORIGINS", "http://127.0.0.1:5173,http://localhost:5173")
    cors_origins = tuple(item.strip() for item in origins_raw.split(",") if item.strip())

    return Settings(
        app_name="Adaptive Digital Literacy Trainer",
        api_version="2.0.0",
        database_url=os.getenv("ADAPTIVE_DATABASE_URL", f"sqlite:///{data_dir / 'adaptive_diglit_ml.db'}"),
        jwt_secret=os.getenv("ADAPTIVE_JWT_SECRET", "change-this-secret-before-production"),
        jwt_algorithm="HS256",
        access_token_minutes=int(os.getenv("ADAPTIVE_ACCESS_TOKEN_MINUTES", "60")),
        refresh_token_days=int(os.getenv("ADAPTIVE_REFRESH_TOKEN_DAYS", "14")),
        model_dir=model_dir,
        report_dir=report_dir,
        target_success_probability=float(os.getenv("ADAPTIVE_TARGET_SUCCESS", "0.70")),
        minimum_training_rows=int(os.getenv("ADAPTIVE_MIN_TRAINING_ROWS", "250")),
        random_seed=int(os.getenv("ADAPTIVE_RANDOM_SEED", "42")),
        cors_origins=cors_origins,
    )
