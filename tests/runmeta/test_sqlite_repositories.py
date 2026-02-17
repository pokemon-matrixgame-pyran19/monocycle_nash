from monocycle_nash.runmeta.clock import now_jst_iso
from monocycle_nash.runmeta.db import SQLiteConnectionFactory, migrate
from monocycle_nash.runmeta.repositories import ProjectsRepository, RunsRepository


def test_connection_enables_foreign_keys() -> None:
    conn = SQLiteConnectionFactory(":memory:").connect()
    assert conn.execute("PRAGMA foreign_keys;").fetchone()[0] == 1


def test_migration_and_repository_flow() -> None:
    conn = SQLiteConnectionFactory(":memory:").connect()
    migrate(conn)
    projects = ProjectsRepository(conn)
    runs = RunsRepository(conn)

    projects.add(
        project_id="analysis-main",
        project_path="C:/analysis/main",
        created_at=now_jst_iso(),
        note="baseline",
    )

    now = now_jst_iso()
    run_id = runs.create_running(
        command="uv run python -m monocycle_nash",
        git_commit="abc123",
        note="first",
        project_id="analysis-main",
        started_at=now,
        created_at=now,
        updated_at=now,
    )
    runs.finish(run_id=run_id, status="success", ended_at=now_jst_iso(), updated_at=now_jst_iso())

    row = runs.find_by_id(run_id)
    assert row is not None
    assert row.status == "success"

    listed = runs.list_runs(status="success", project_id="analysis-main")
    assert [r.run_id for r in listed] == [run_id]


def test_projects_repository_list_projects() -> None:
    conn = SQLiteConnectionFactory(":memory:").connect()
    migrate(conn)
    projects = ProjectsRepository(conn)

    projects.add(project_id="a", project_path="C:/a", created_at="2024-01-01T00:00:00+09:00", note="")
    projects.add(project_id="b", project_path="C:/b", created_at="2024-01-02T00:00:00+09:00", note="note")

    rows = projects.list_projects()

    assert [row.project_id for row in rows] == ["b", "a"]
