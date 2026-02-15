import json
from pathlib import Path

from monocycle_nash.runmeta.artifact_store import RunArtifactStore
from monocycle_nash.runmeta.db import SQLiteConnectionFactory, migrate
from monocycle_nash.runmeta.repositories import ProjectsRepository, RunsRepository
from monocycle_nash.runmeta.service import RunSessionService


def test_run_artifact_store_create_write_snapshot_delete(tmp_path: Path) -> None:
    source_a = tmp_path / "input_a.toml"
    source_a.write_text("x = 1\n", encoding="utf-8")
    source_b = tmp_path / "input_a.toml.copy"
    source_b.write_text("y = 2\n", encoding="utf-8")

    store = RunArtifactStore(tmp_path / "result")
    run_id = 101

    store.create_run_dir(run_id)
    store.write_input_file(run_id, "raw.toml", "k = 3\n")
    store.write_output_file(run_id, "result.json", '{"ok": true}')
    store.write_log_file(run_id, "run.log", "started")
    store.write_meta(run_id, {"run_id": run_id, "status": "running"})
    copied = store.save_input_snapshot(
        run_id,
        [source_a, source_b],
        merged_input_content="z = 9\n",
    )

    run_dir = tmp_path / "result" / str(run_id)
    assert (run_dir / "input" / "raw.toml").exists()
    assert (run_dir / "output" / "result.json").exists()
    assert (run_dir / "logs" / "run.log").exists()
    assert (run_dir / "meta.json").exists()
    assert copied[-1].name == "merged_input.toml"
    assert (run_dir / "input" / "merged_input.toml").read_text(encoding="utf-8") == "z = 9\n"

    store.delete_run_artifacts(run_id)
    assert not run_dir.exists()


def test_run_session_service_start_and_finish_updates_db_and_meta(tmp_path: Path) -> None:
    conn = SQLiteConnectionFactory(":memory:").connect()
    migrate(conn)

    projects = ProjectsRepository(conn)
    runs = RunsRepository(conn)
    project_id = projects.add("alpha")

    artifacts = RunArtifactStore(tmp_path / "result")
    service = RunSessionService(run_repository=runs, artifact_store=artifacts)

    run_id = service.start(project_id)
    start_record = runs.find_by_id(run_id)
    assert start_record is not None
    assert start_record["status"] == "running"

    start_meta = json.loads((tmp_path / "result" / str(run_id) / "meta.json").read_text(encoding="utf-8"))
    assert start_meta["status"] == "running"

    service.finish_fail(run_id, {"error": "boom"})
    finish_record = runs.find_by_id(run_id)
    assert finish_record is not None
    assert finish_record["status"] == "fail"
    assert finish_record["ended_at"] is not None

    finish_meta = json.loads((tmp_path / "result" / str(run_id) / "meta.json").read_text(encoding="utf-8"))
    assert finish_meta["status"] == "fail"
    assert finish_meta["details"]["error"] == "boom"


def test_run_session_service_start_rolls_back_on_artifact_failure(tmp_path: Path) -> None:
    class FailingArtifactStore(RunArtifactStore):
        def write_meta(self, run_id: int | str, meta: dict[str, object]):  # type: ignore[override]
            super().create_run_dir(run_id)
            raise OSError("meta write failed")

    conn = SQLiteConnectionFactory(":memory:").connect()
    migrate(conn)

    projects = ProjectsRepository(conn)
    runs = RunsRepository(conn)
    project_id = projects.add("alpha")

    artifacts = FailingArtifactStore(tmp_path / "result")
    service = RunSessionService(run_repository=runs, artifact_store=artifacts)

    try:
        service.start(project_id)
    except OSError as exc:
        assert str(exc) == "meta write failed"
    else:
        raise AssertionError("start() should raise when artifact creation fails")

    assert runs.list_runs(project_id=project_id) == []
    result_root = tmp_path / "result"
    assert not result_root.exists() or list(result_root.iterdir()) == []
