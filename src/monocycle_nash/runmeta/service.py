"""Application service for run metadata operations."""

from __future__ import annotations

from dataclasses import dataclass

from .artifact_store import ArtifactStore
from .clock import now_jst_iso
from .models import RunStatus
from .repositories import ProjectsRepository, RunsRepository


@dataclass
class RunMetaService:
    """Coordinates project/run repositories and artifact storage."""

    project_repository: ProjectsRepository
    run_repository: RunsRepository
    artifact_store: ArtifactStore

    def register_project(self, name: str) -> int:
        """Create a project and return its id."""

        return self.project_repository.add(name=name)

    def start_run(self, project_id: int) -> int:
        """Start a run for a project and return run id."""

        return self.run_repository.create_running(project_id=project_id)

    def finish_run(self, run_id: int, status: RunStatus) -> None:
        """Finish a run with a terminal status."""

        self.run_repository.finish(run_id=run_id, status=status)

    def attach_artifact(self, run_id: str, artifact_uri: str) -> None:
        """Attach an artifact URI to a run."""

        self.artifact_store.add(run_id=run_id, artifact_uri=artifact_uri)

    def health(self) -> dict[str, str]:
        """Return a lightweight service status payload."""

        return {"status": "ok", "timestamp": now_jst_iso()}
