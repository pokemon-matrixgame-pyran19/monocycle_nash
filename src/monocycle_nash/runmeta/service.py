"""Application services for run metadata operations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .artifact_store import RunArtifactStore
from .clock import now_jst_iso
from .models import RunStatus
from .repositories import ProjectsRepository, RunsRepository


@dataclass
class RunSessionService:
    """Coordinate DB run status changes with run artifact lifecycle."""

    run_repository: RunsRepository
    artifact_store: RunArtifactStore

    def start(self, project_id: int) -> int:
        """Create DB run row, allocate run dir and write initial meta.json."""

        run_id = self.run_repository.create_running(project_id=project_id)
        record = self.run_repository.find_by_id(run_id)
        if record is None:
            raise RuntimeError(f"Failed to load created run record: run_id={run_id}")

        try:
            self.artifact_store.create_run_dir(run_id)
            self.artifact_store.write_meta(run_id, self._build_meta(record))
            return run_id
        except Exception:
            self._rollback_start(run_id)
            raise

    def _rollback_start(self, run_id: int) -> None:
        """Rollback DB/artifact side effects when start lifecycle fails."""

        try:
            self.run_repository.delete(run_id)
        finally:
            self.artifact_store.delete_run_artifacts(run_id)

    def finish_success(self, run_id: int, details: Mapping[str, object] | None = None) -> None:
        self._finish(run_id, "success", details)

    def finish_fail(self, run_id: int, details: Mapping[str, object] | None = None) -> None:
        self._finish(run_id, "fail", details)

    def finish_killed(self, run_id: int, details: Mapping[str, object] | None = None) -> None:
        self._finish(run_id, "killed", details)

    def _finish(
        self,
        run_id: int,
        status: RunStatus,
        details: Mapping[str, object] | None,
    ) -> None:
        self.run_repository.finish(run_id=run_id, status=status)
        record = self.run_repository.find_by_id(run_id)
        if record is None:
            raise RuntimeError(f"Run record not found after finish: run_id={run_id}")

        meta = self._build_meta(record)
        if details:
            meta["details"] = dict(details)
        self.artifact_store.write_meta(run_id, meta)

    @staticmethod
    def _build_meta(record: Mapping[str, object]) -> dict[str, object]:
        return {
            "run_id": record["id"],
            "project_id": record["project_id"],
            "status": record["status"],
            "created_at": record["created_at"],
            "ended_at": record["ended_at"],
            "updated_at": record["updated_at"],
            "meta_written_at": now_jst_iso(),
        }


@dataclass
class RunMetaService:
    """Coordinates project/run repositories and artifact storage."""

    project_repository: ProjectsRepository
    run_repository: RunsRepository
    artifact_store: RunArtifactStore

    def register_project(self, name: str) -> int:
        return self.project_repository.add(name=name)

    def start_run(self, project_id: int) -> int:
        return self.run_repository.create_running(project_id=project_id)

    def finish_run(self, run_id: int, status: RunStatus) -> None:
        self.run_repository.finish(run_id=run_id, status=status)

    def health(self) -> dict[str, str]:
        return {"status": "ok", "timestamp": now_jst_iso()}
