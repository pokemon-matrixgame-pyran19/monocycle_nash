"""CLI entry helpers for run metadata service."""

from __future__ import annotations

from .artifact_store import ArtifactStore
from .db import SQLiteConnectionFactory, migrate
from .repositories import ProjectRepository, RunRepository
from .service import RunMetaService


def build_default_service() -> RunMetaService:
    """Create a default in-memory SQLite-backed RunMetaService instance."""

    conn = SQLiteConnectionFactory(":memory:").connect()
    migrate(conn)
    return RunMetaService(
        project_repository=ProjectRepository(conn=conn),
        run_repository=RunRepository(conn=conn),
        artifact_store=ArtifactStore(),
    )
