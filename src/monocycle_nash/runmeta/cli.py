"""Argparse based CLI for run metadata management."""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path
from typing import Sequence, cast

from .artifact_store import RunArtifactStore
from .db import SQLiteConnectionFactory, migrate
from .models import RunStatus
from .repositories import ProjectsRepository, RunsRepository
from .service import RunMetaService

DEFAULT_DB_PATH = Path(".runmeta/run_history.db")


def build_default_service() -> RunMetaService:
    """Create a default in-memory SQLite-backed RunMetaService instance."""

    conn = SQLiteConnectionFactory(":memory:").connect()
    migrate(conn)
    return RunMetaService(
        project_repository=ProjectsRepository(conn=conn),
        run_repository=RunsRepository(conn=conn),
        artifact_store=RunArtifactStore(),
    )


def _open_repositories(db_path: str | Path) -> tuple[sqlite3.Connection, ProjectsRepository, RunsRepository]:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = SQLiteConnectionFactory(path).connect()
    migrate(conn)
    return conn, ProjectsRepository(conn), RunsRepository(conn)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="runmeta")
    parser.add_argument("--db-path", default=str(DEFAULT_DB_PATH), help="Path to SQLite DB")
    subparsers = parser.add_subparsers(dest="command", required=True)

    update_run = subparsers.add_parser("update-run", help="Update an existing run")
    update_run.add_argument("--run-id", type=int, required=True)
    update_run.add_argument("--status", choices=["running", "success", "fail", "killed"])
    update_run.add_argument("--ended-at")
    update_run.add_argument("--project-id", type=int)

    delete_run = subparsers.add_parser("delete-run", help="Delete an existing run")
    delete_run.add_argument("--run-id", type=int, required=True)

    add_project = subparsers.add_parser("add-project", help="Add a project")
    add_project.add_argument("--name", required=True)

    update_project = subparsers.add_parser("update-project", help="Update an existing project")
    update_project.add_argument("--project-id", type=int, required=True)
    update_project.add_argument("--name", required=True)

    delete_project = subparsers.add_parser("delete-project", help="Delete an existing project")
    delete_project.add_argument("--project-id", type=int, required=True)

    list_runs = subparsers.add_parser("list-runs", help="List runs with optional filters")
    list_runs.add_argument("--from", dest="from_at")
    list_runs.add_argument("--to", dest="to_at")
    list_runs.add_argument("--status", choices=["running", "success", "fail", "killed"])
    list_runs.add_argument("--project-id", type=int)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    conn, projects, runs = _open_repositories(args.db_path)
    try:
        if args.command == "update-run":
            if args.status is None and args.ended_at is None and args.project_id is None:
                parser.error("update-run requires at least one of --status, --ended-at, --project-id")
            if runs.find_by_id(args.run_id) is None:
                print(f"run id {args.run_id} not found")
                return 1
            if args.project_id is not None and projects.find(args.project_id) is None:
                print(f"project id {args.project_id} not found")
                return 1
            runs.update(
                args.run_id,
                status=cast(RunStatus | None, args.status),
                ended_at=args.ended_at,
                project_id=args.project_id,
            )
            print(f"updated run {args.run_id}")
            return 0

        if args.command == "delete-run":
            if runs.find_by_id(args.run_id) is None:
                print(f"run id {args.run_id} not found")
                return 1
            runs.delete(args.run_id)
            print(f"deleted run {args.run_id}")
            return 0

        if args.command == "add-project":
            project_id = projects.add(args.name)
            print(project_id)
            return 0

        if args.command == "update-project":
            if projects.find(args.project_id) is None:
                print(f"project id {args.project_id} not found")
                return 1
            projects.update(args.project_id, name=args.name)
            print(f"updated project {args.project_id}")
            return 0

        if args.command == "delete-project":
            if projects.find(args.project_id) is None:
                print(f"project id {args.project_id} not found")
                return 1
            projects.delete(args.project_id)
            print(f"deleted project {args.project_id}")
            return 0

        if args.command == "list-runs":
            if args.project_id is not None and projects.find(args.project_id) is None:
                print(f"project id {args.project_id} not found")
                return 1
            rows = runs.list_runs(
                from_at=args.from_at,
                to_at=args.to_at,
                status=args.status,
                project_id=args.project_id,
            )
            print("id\tproject_id\tstatus\tcreated_at\tended_at\tupdated_at")
            for row in rows:
                print(
                    f"{row['id']}\t{row['project_id']}\t{row['status']}\t"
                    f"{row['created_at']}\t{row['ended_at'] or ''}\t{row['updated_at']}"
                )
            return 0

        parser.error(f"unknown command: {args.command}")
    finally:
        conn.close()

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
