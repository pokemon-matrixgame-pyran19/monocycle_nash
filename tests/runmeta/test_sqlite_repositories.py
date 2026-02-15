from monocycle_nash.runmeta.db import SQLiteConnectionFactory, migrate
from monocycle_nash.runmeta.repositories import ProjectsRepository, RunsRepository


def test_connection_enables_foreign_keys() -> None:
    conn = SQLiteConnectionFactory(":memory:").connect()
    pragma = conn.execute("PRAGMA foreign_keys;").fetchone()
    assert pragma is not None
    assert pragma[0] == 1


def test_migrate_and_repository_crud_and_filters() -> None:
    conn = SQLiteConnectionFactory(":memory:").connect()
    migrate(conn)
    projects = ProjectsRepository(conn)
    runs = RunsRepository(conn)

    project_a = projects.add("alpha")
    project_b = projects.add("beta")

    run_a = runs.create_running(project_a)
    run_b = runs.create_running(project_b)

    runs.finish(run_a, "success")
    runs.update(run_b, status="fail")

    fetched = runs.find_by_id(run_a)
    assert fetched is not None
    assert fetched["status"] == "success"
    assert fetched["ended_at"] is not None

    success_runs = runs.list_runs(status="success")
    assert [run["id"] for run in success_runs] == [run_a]

    filtered = runs.list_runs(project_id=project_b, status="fail")
    assert [run["id"] for run in filtered] == [run_b]

    projects.delete(project_b)
    assert runs.find_by_id(run_b) is None
