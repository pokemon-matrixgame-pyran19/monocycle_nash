"""Repository layer for run metadata records."""

from __future__ import annotations

from dataclasses import dataclass

from .clock import now_jst_iso
from .db import RunMetaDB
from .models import ProjectRecord, RunRecord, RunStatus


@dataclass
class ProjectRepository:
    """Access and manage project records."""

    db: RunMetaDB

    def save(self, project_id: str, name: str) -> ProjectRecord:
        """Create or update a project record."""

        existing = self.db.projects.get(project_id)
        timestamp = now_jst_iso()
        record = ProjectRecord(
            project_id=project_id,
            name=name,
            created_at=existing.created_at if existing else timestamp,
            updated_at=timestamp,
        )
        self.db.projects[project_id] = record
        return record

    def get(self, project_id: str) -> ProjectRecord | None:
        """Fetch a project by id."""

        return self.db.projects.get(project_id)


@dataclass
class RunRepository:
    """Access and manage run records."""

    db: RunMetaDB

    def create(self, run_id: str, project_id: str) -> RunRecord:
        """Create a running run record."""

        record = RunRecord(
            run_id=run_id,
            project_id=project_id,
            status="running",
            started_at=now_jst_iso(),
        )
        self.db.runs[run_id] = record
        return record

    def update_status(self, run_id: str, status: RunStatus) -> RunRecord:
        """Update status and completion timestamp for a run record."""

        current = self.db.runs[run_id]
        ended_at = now_jst_iso() if status != "running" else None
        updated = RunRecord(
            run_id=current.run_id,
            project_id=current.project_id,
            status=status,
            started_at=current.started_at,
            ended_at=ended_at,
        )
        self.db.runs[run_id] = updated
        return updated

    def get(self, run_id: str) -> RunRecord | None:
        """Fetch a run by id."""

        return self.db.runs.get(run_id)
