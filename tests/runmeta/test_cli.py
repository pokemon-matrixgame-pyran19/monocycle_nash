from pathlib import Path

from monocycle_nash.runmeta.cli import main
from monocycle_nash.runmeta.db import SQLiteConnectionFactory, migrate
from monocycle_nash.runmeta.repositories import ProjectsRepository, RunsRepository


def _seed(db_path: Path) -> tuple[int, int, int]:
    conn = SQLiteConnectionFactory(db_path).connect()
    migrate(conn)
    projects = ProjectsRepository(conn)
    runs = RunsRepository(conn)

    project_a = projects.add("alpha")
    project_b = projects.add("beta")
    run_a = runs.create_running(project_a)
    run_b = runs.create_running(project_b)
    runs.update(run_a, status="success")
    runs.update(run_b, status="fail")

    conn.execute(
        "UPDATE runs SET created_at = ? WHERE id = ?",
        ("2026-01-01T12:00:00+09:00", run_a),
    )
    conn.execute(
        "UPDATE runs SET created_at = ? WHERE id = ?",
        ("2026-01-02T12:00:00+09:00", run_b),
    )
    conn.commit()
    conn.close()

    return project_a, project_b, run_a


def test_update_run_nonexistent_id_returns_non_zero(tmp_path: Path, capsys) -> None:
    db_path = tmp_path / "run_history.db"

    code = main(["--db-path", str(db_path), "update-run", "--run-id", "999", "--status", "success"])

    assert code == 1
    assert "run id 999 not found" in capsys.readouterr().out


def test_delete_project_nonexistent_id_returns_non_zero(tmp_path: Path, capsys) -> None:
    db_path = tmp_path / "run_history.db"

    code = main(["--db-path", str(db_path), "delete-project", "--project-id", "123"])

    assert code == 1
    assert "project id 123 not found" in capsys.readouterr().out


def test_list_runs_with_filters_and_descending_created_at(tmp_path: Path, capsys) -> None:
    db_path = tmp_path / "run_history.db"
    _, project_b, _ = _seed(db_path)

    code = main(
        [
            "--db-path",
            str(db_path),
            "list-runs",
            "--status",
            "fail",
            "--project-id",
            str(project_b),
        ]
    )

    assert code == 0
    output_lines = capsys.readouterr().out.strip().splitlines()
    assert output_lines[0].startswith("id\tproject_id\tstatus")
    assert len(output_lines) == 2
    assert "\tfail\t" in output_lines[1]
    assert output_lines[1].split("\t")[1] == str(project_b)


def test_list_runs_outputs_created_at_descending(tmp_path: Path, capsys) -> None:
    db_path = tmp_path / "run_history.db"
    _, _, _ = _seed(db_path)

    code = main(["--db-path", str(db_path), "list-runs"])

    assert code == 0
    output_lines = capsys.readouterr().out.strip().splitlines()
    created_values = [line.split("\t")[3] for line in output_lines[1:]]
    assert created_values == sorted(created_values, reverse=True)
