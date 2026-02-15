"""CLI entry helpers for run metadata service."""

from __future__ import annotations

from .artifact_store import ArtifactStore
from .db import RunMetaDB
from .service import RunMetaService
from .repositories import ProjectRepository, RunRepository


def build_default_service() -> RunMetaService:
    """Create a default in-memory RunMetaService instance."""

    db = RunMetaDB()
    return RunMetaService(
        project_repository=ProjectRepository(db=db),
        run_repository=RunRepository(db=db),
        artifact_store=ArtifactStore(),
    )
