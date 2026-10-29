from __future__ import annotations

from datetime import datetime, timezone


def utcnow_naive() -> datetime:
    """Return UTC time without tzinfo for compatibility with the current SQLite schema."""
    return datetime.now(timezone.utc).replace(tzinfo=None)
