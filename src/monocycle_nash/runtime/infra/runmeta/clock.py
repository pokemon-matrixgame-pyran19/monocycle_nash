"""Time utilities for run metadata modules."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone

JST = timezone(timedelta(hours=9), name="JST")


def now_jst_iso() -> str:
    """Return the current JST timestamp in ISO-8601 format."""

    return datetime.now(tz=UTC).astimezone(JST).isoformat()
