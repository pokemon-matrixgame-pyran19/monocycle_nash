"""Run metadata domain models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

RunStatus = Literal["running", "success", "fail", "killed"]


@dataclass(frozen=True)
class ProjectRecord:
    """Represents a project that owns one or more run records."""

    project_id: str
    name: str
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class RunRecord:
    """Represents one execution run in the metadata store."""

    run_id: str
    project_id: str
    status: RunStatus
    started_at: str
    ended_at: str | None = None
