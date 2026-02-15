import json
from pathlib import Path

import pytest

from monocycle_nash.runmeta.artifact_store import RunArtifactStore
from monocycle_nash.runmeta.db import SQLiteConnectionFactory, migrate
from monocycle_nash.runmeta.repositories import ProjectsRepository, RunsRepository
from monocycle_nash.runmeta.service import RunSessionService


def test_artifact_store_snapshot_and_meta(tmp_path: Path) -> None:
    src = tmp_path / "input.toml"
    src.write_text("x=1\n", encoding="utf-8")

    store = RunArtifactStore(tmp_path / "result")
    info = store.save_input_snapshot(1, src, merged_input_bytes=b"m=1\n")
    assert info.split_config_applied is True
    assert info.stored_file == "input/merged_input.toml"

    store.write_initial_meta(1, {"run_id": 1, "status": "running"})
    meta = json.loads((tmp_path / "result" / "1" / "meta.json").read_text(encoding="utf-8"))
    assert meta["status"] == "running"


def test_run_session_service_start_finish_with_input_snapshot(tmp_path: Path) -> None:
    conn = SQLiteConnectionFactory(":memory:").connect()
    migrate(conn)
    projects = ProjectsRepository(conn)
    runs = RunsRepository(conn)
    projects.add(project_id="p1", project_path="C:/p1", created_at="2026-01-01T00:00:00+09:00")

    src_input = tmp_path / "input.toml"
    src_input.write_text("a=1\n", encoding="utf-8")

    service = RunSessionService(runs_repository=runs, artifact_store=RunArtifactStore(tmp_path / "result"))
    ctx = service.start(
        command="uv run app",
        project_id="p1",
        note="n",
        src_input_path=src_input,
        merged_input_bytes=b"merged=1\n",
    )
    assert runs.find_by_id(ctx.run_id) is not None

    start_meta = json.loads((tmp_path / "result" / str(ctx.run_id) / "meta.json").read_text(encoding="utf-8"))
    assert start_meta["input"]["split_config_applied"] is True

    service.finish_fail(ctx, extra_meta={"reason": "boom"})
    ended = runs.find_by_id(ctx.run_id)
    assert ended is not None
    assert ended.status == "fail"

    finish_meta = json.loads((tmp_path / "result" / str(ctx.run_id) / "meta.json").read_text(encoding="utf-8"))
    assert finish_meta["reason"] == "boom"


def test_run_session_service_start_rolls_back_when_meta_write_fails(tmp_path: Path) -> None:
    class FailingStore(RunArtifactStore):
        def write_initial_meta(self, run_id: int, payload: dict[str, object]) -> None:  # type: ignore[override]
            raise OSError("meta write failed")

    conn = SQLiteConnectionFactory(":memory:").connect()
    migrate(conn)
    projects = ProjectsRepository(conn)
    runs = RunsRepository(conn)
    projects.add(project_id="p1", project_path="C:/p1", created_at="2026-01-01T00:00:00+09:00")

    service = RunSessionService(runs_repository=runs, artifact_store=FailingStore(tmp_path / "result"))
    with pytest.raises(OSError, match="meta write failed"):
        service.start(command="uv run app", project_id="p1")

    assert runs.list_runs(project_id="p1") == []
