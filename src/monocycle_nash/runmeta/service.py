"""Application service for run metadata operations."""

from __future__ import annotations

from dataclasses import dataclass

from .artifact_store import ArtifactStore
from .clock import now_jst_iso
from .models import ProjectRecord, RunRecord, RunStatus
from .repositories import ProjectRepository, RunRepository


@dataclass
class RunMetaService:
    """Coordinates project/run repositories and artifact storage."""

    project_repository: ProjectRepository
    run_repository: RunRepository
    artifact_store: ArtifactStore

    def register_project(self, project_id: str, name: str) -> ProjectRecord:
        """Create or update a project."""

        return self.project_repository.save(project_id=project_id, name=name)

    def start_run(self, run_id: str, project_id: str) -> RunRecord:
        """Start a run for a project."""

        return self.run_repository.create(run_id=run_id, project_id=project_id)

    def finish_run(self, run_id: str, status: RunStatus) -> RunRecord:
        """Finish a run with a terminal status."""

        return self.run_repository.update_status(run_id=run_id, status=status)

    def attach_artifact(self, run_id: str, artifact_uri: str) -> None:
        """Attach an artifact URI to a run."""

        self.artifact_store.add(run_id=run_id, artifact_uri=artifact_uri)

    def health(self) -> dict[str, str]:
        """Return a lightweight service status payload."""

        return {"status": "ok", "timestamp": now_jst_iso()}
