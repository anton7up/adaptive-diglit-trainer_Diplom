from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import get_settings
from .database import Base, SessionLocal, engine
from .routers import admin, analytics, auth, catalog, exams, ml, practice, research, teacher
from .services.seed import seed_catalog


@asynccontextmanager
async def lifespan(_: FastAPI):
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        seed_catalog(db)
    finally:
        db.close()
    yield


settings = get_settings()
app = FastAPI(
    title=f"{settings.app_name} API",
    version=settings.api_version,
    description="ML-driven adaptive training platform for digital literacy.",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_origins),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(admin.router)
app.include_router(catalog.router)
app.include_router(practice.router)
app.include_router(analytics.router)
app.include_router(exams.router)
app.include_router(ml.router)
app.include_router(research.router)
app.include_router(teacher.router)


@app.get("/health")
def health():
    return {"status": "ok", "version": settings.api_version}
