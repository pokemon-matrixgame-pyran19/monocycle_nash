"""Use-case service that orchestrates DB-first run lifecycle."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .artifact_store import RunArtifactStore
from .clock import now_jst_iso
from .models import RunContext, RunStatus
from .repositories import RunsRepository


@dataclass
class RunSessionService:
    runs_repository: RunsRepository
    artifact_store: RunArtifactStore

    def start(
        self,
        *,
        command: str,
        project_id: str | None = None,
        note: str = "",
        git_commit: str | None = None,
        src_input_path: str | Path | None = None,
        merged_input_bytes: bytes | None = None,
    ) -> RunContext:
        now = now_jst_iso()
        run_id = self.runs_repository.create_running(
            command=command,
            git_commit=git_commit,
            note=note,
            project_id=project_id,
            started_at=now,
            created_at=now,
            updated_at=now,
        )
        try:
            self.artifact_store.create_run_dir(run_id)
            input_info = None
            if src_input_path is not None:
                input_info = self.artifact_store.save_input_snapshot(
                    run_id,
                    src_input_path,
                    merged_input_bytes=merged_input_bytes,
                )

            record = self.runs_repository.find_by_id(run_id)
            if record is None:
                raise RuntimeError(f"run not found after create: {run_id}")

            payload = self._meta_payload(record)
            if input_info is not None:
                payload["input"] = {
                    "source_path": input_info.source_path,
                    "split_config_applied": input_info.split_config_applied,
                    "stored_file": input_info.stored_file,
                    "checksum_sha256": input_info.checksum_sha256,
                }
            self.artifact_store.write_initial_meta(run_id, payload)
            return RunContext(run_id=run_id, result_dir=str(self.artifact_store.run_dir(run_id)))
        except Exception:
            self.runs_repository.delete(run_id)
            self.artifact_store.delete_run_dir(run_id)
            raise

    def finish(self, ctx: RunContext, *, status: RunStatus, extra_meta: dict[str, Any] | None = None) -> None:
        now = now_jst_iso()
        self.runs_repository.finish(run_id=ctx.run_id, status=status, ended_at=now, updated_at=now)
        record = self.runs_repository.find_by_id(ctx.run_id)
        if record is None:
            raise RuntimeError(f"run not found after finish: {ctx.run_id}")
        payload = self._meta_payload(record)
        if extra_meta is not None:
            payload.update(extra_meta)
        self.artifact_store.write_final_meta(ctx.run_id, payload)

    def finish_success(self, ctx: RunContext, extra_meta: dict[str, Any] | None = None) -> None:
        self.finish(ctx, status="success", extra_meta=extra_meta)

    def finish_fail(self, ctx: RunContext, extra_meta: dict[str, Any] | None = None) -> None:
        self.finish(ctx, status="fail", extra_meta=extra_meta)

    def finish_killed(self, ctx: RunContext, extra_meta: dict[str, Any] | None = None) -> None:
        self.finish(ctx, status="killed", extra_meta=extra_meta)

    @staticmethod
    def _meta_payload(record: Any) -> dict[str, Any]:
        return {
            "run_id": record.run_id,
            "created_at": record.created_at,
            "started_at": record.started_at,
            "ended_at": record.ended_at,
            "status": record.status,
            "command": record.command,
            "git_commit": record.git_commit,
            "note": record.note,
            "project_id": record.project_id,
            "project_path": record.project_path,
            "updated_at": record.updated_at,
        }
