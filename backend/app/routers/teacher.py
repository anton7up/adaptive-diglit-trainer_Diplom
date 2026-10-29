from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from .. import models
from ..database import get_db
from ..dependencies import require_roles
from ..services.item_analysis import analyze_item_bank, summarize_item_bank

router = APIRouter(prefix="/teacher", tags=["teacher"])


@router.get("/items/analysis")
def item_analysis(
    include_synthetic: bool = False,
    db: Session = Depends(get_db),
    _: models.User = Depends(require_roles("teacher", "admin")),
):
    rows = analyze_item_bank(db, include_synthetic=include_synthetic)
    return {"summary": summarize_item_bank(rows), "items": rows}
