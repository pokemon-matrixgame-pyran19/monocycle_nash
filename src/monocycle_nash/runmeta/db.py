"""In-memory persistence primitives for run metadata."""

from __future__ import annotations

from dataclasses import dataclass, field

from .models import ProjectRecord, RunRecord


@dataclass
class RunMetaDB:
    """Simple in-memory database for projects and runs."""

    projects: dict[str, ProjectRecord] = field(default_factory=dict)
    runs: dict[str, RunRecord] = field(default_factory=dict)
