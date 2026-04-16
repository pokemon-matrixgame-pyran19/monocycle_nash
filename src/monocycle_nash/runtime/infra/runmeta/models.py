"""Run metadata data models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

RunStatus = Literal["running", "success", "fail", "killed"]


@dataclass(frozen=True)
class ProjectRecord:
    project_id: str
    project_path: str
    created_at: str
    note: str = ""


@dataclass(frozen=True)
class RunRecord:
    run_id: int
    created_at: str
    started_at: str
    ended_at: str | None
    status: RunStatus
    command: str
    git_commit: str | None
    note: str
    project_id: str | None
    project_path: str | None
    updated_at: str


@dataclass(frozen=True)
class InputSnapshotInfo:
    source_path: str
    split_config_applied: bool
    stored_file: str | None
    checksum_sha256: str | None


@dataclass(frozen=True)
class RunContext:
    run_id: int
    result_dir: str
