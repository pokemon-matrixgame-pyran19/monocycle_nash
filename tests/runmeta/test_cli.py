from pathlib import Path

from monocycle_nash.runmeta.artifact_store import RunArtifactStore
from monocycle_nash.runmeta.cli import main
from monocycle_nash.runmeta.clock import now_jst_iso
from monocycle_nash.runmeta.db import SQLiteConnectionFactory, migrate
from monocycle_nash.runmeta.repositories import UNASSIGNED_PROJECT_ID, ProjectsRepository, RunsRepository


def _seed(db_path: Path) -> int:
    conn = SQLiteConnectionFactory(db_path).connect()
    migrate(conn)
    projects = ProjectsRepository(conn)
    runs = RunsRepository(conn)
    projects.add(project_id="a", project_path="C:/a", created_at=now_jst_iso())
    now = now_jst_iso()
    run_id = runs.create_running(
        command="cmd",
        git_commit=None,
        note="",
        project_id="a",
        started_at=now,
        created_at=now,
        updated_at=now,
    )
    runs.update(run_id, status="fail", updated_at=now_jst_iso())
    conn.close()
    return run_id


def test_cli_not_found_returns_non_zero(tmp_path: Path, capsys) -> None:
    db = tmp_path / "db.sqlite"
    code = main(["--db-path", str(db), "delete-project", "--project-id", "x"])
    assert code == 1
    assert "not found" in capsys.readouterr().out


def test_cli_list_runs(tmp_path: Path, capsys) -> None:
    db = tmp_path / "db.sqlite"
    _seed(db)
    code = main(["--db-path", str(db), "list-runs", "--status", "fail", "--project-id", "a"])
    assert code == 0
    out = capsys.readouterr().out
    assert "run_id" in out
    assert "\tfail\t" in out
    created_at = out.strip().splitlines()[1].split("\t")[-1]
    assert len(created_at) == len("2000-01-01 00:00")
    assert "+" not in created_at


def test_cli_delete_run_with_files_removes_artifacts(tmp_path: Path, capsys, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    db = tmp_path / "db.sqlite"
    run_id = _seed(db)
    run_dir = RunArtifactStore().run_dir(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)

    code = main(["--db-path", str(db), "delete-run", "--run-id", str(run_id), "--with-files"])

    assert code == 0
    assert "deleted run" in capsys.readouterr().out
    assert not run_dir.exists()


def test_cli_list_projects(tmp_path: Path, capsys) -> None:
    db = tmp_path / "db.sqlite"
    _seed(db)

    code = main(["--db-path", str(db), "list-projects"])

    assert code == 0
    out = capsys.readouterr().out
    assert "project_id" in out
    assert "a	C:/a" in out
    created_at = out.strip().splitlines()[1].split("\t")[2]
    assert len(created_at) == len("2000-01-01 00:00")
    assert "+" not in created_at


def test_cli_list_runs_with_unassigned_project_filter(tmp_path: Path, capsys) -> None:
    db = tmp_path / "db.sqlite"
    conn = SQLiteConnectionFactory(db).connect()
    migrate(conn)
    runs = RunsRepository(conn)
    now = now_jst_iso()
    runs.create_running(
        command="cmd",
        git_commit=None,
        note="",
        project_id=None,
        started_at=now,
        created_at=now,
        updated_at=now,
    )
    conn.close()

    code = main(["--db-path", str(db), "list-runs", "--project-id", UNASSIGNED_PROJECT_ID])

    assert code == 0
    out = capsys.readouterr().out
    assert "run_id" in out
    assert "\trunning\t" in out


def test_cli_regenerate_project_refs(tmp_path: Path, capsys) -> None:
    db = tmp_path / "db.sqlite"
    project_dir = tmp_path / "project"
    conn = SQLiteConnectionFactory(db).connect()
    migrate(conn)
    projects = ProjectsRepository(conn)
    runs = RunsRepository(conn)

    projects.add(project_id="a", project_path=str(project_dir), created_at=now_jst_iso())
    now = now_jst_iso()
    run_id = runs.create_running(
        command="cmd",
        git_commit=None,
        note="",
        project_id="a",
        started_at=now,
        created_at=now,
        updated_at=now,
    )
    conn.close()

    code = main(
        [
            "--db-path",
            str(db),
            "regenerate-project-refs",
            "--project-id",
            "a",
            "--result-base-dir",
            str(tmp_path / "results"),
        ]
    )

    assert code == 0
    assert "regenerated 1 references" in capsys.readouterr().out

    refs_dir = project_dir / "experiment_refs"
    ref_link = refs_dir / str(run_id)
    ref_txt = refs_dir / f"{run_id}.txt"
    assert ref_link.exists() or ref_link.is_symlink() or ref_txt.exists()
