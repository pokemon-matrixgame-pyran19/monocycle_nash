"""CLI for run metadata maintenance operations."""

from __future__ import annotations

import argparse
from pathlib import Path
import sqlite3
from typing import Sequence, cast

from .artifact_store import RunArtifactStore
from .clock import now_jst_iso
from .db import SQLiteConnectionFactory, migrate
from .models import RunStatus
from .repositories import ProjectsRepository, RunsRepository

DEFAULT_DB_PATH = Path(".runmeta/run_history.db")


def _open_repositories(db_path: str | Path) -> tuple[sqlite3.Connection, ProjectsRepository, RunsRepository]:
    conn = SQLiteConnectionFactory(db_path).connect()
    migrate(conn)
    return conn, ProjectsRepository(conn), RunsRepository(conn)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="runmeta")
    parser.add_argument("--db-path", default=str(DEFAULT_DB_PATH))
    sub = parser.add_subparsers(dest="command", required=True)

    u = sub.add_parser("update-run")
    u.add_argument("--run-id", type=int, required=True)
    u.add_argument("--status", choices=["running", "success", "fail", "killed"])
    u.add_argument("--note")
    u.add_argument("--project-id")

    d = sub.add_parser("delete-run")
    d.add_argument("--run-id", type=int, required=True)
    d.add_argument("--with-files", action="store_true")

    ap = sub.add_parser("add-project")
    ap.add_argument("--project-id", required=True)
    ap.add_argument("--project-path", required=True)
    ap.add_argument("--note", default="")

    up = sub.add_parser("update-project")
    up.add_argument("--project-id", required=True)
    up.add_argument("--project-path")
    up.add_argument("--note")

    dp = sub.add_parser("delete-project")
    dp.add_argument("--project-id", required=True)

    sub.add_parser("list-projects")

    l = sub.add_parser("list-runs")
    l.add_argument("--from", dest="from_at")
    l.add_argument("--to", dest="to_at")
    l.add_argument("--status", choices=["running", "success", "fail", "killed"])
    l.add_argument("--project-id")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    conn, projects, runs = _open_repositories(args.db_path)
    artifact_store = RunArtifactStore()

    try:

        if args.command == "update-run":
            if runs.find_by_id(args.run_id) is None:
                print(f"run id {args.run_id} not found")
                return 1
            if args.project_id is not None and projects.find(args.project_id) is None:
                print(f"project id {args.project_id} not found")
                return 1
            if args.status is None and args.note is None and args.project_id is None:
                print("update-run requires at least one of --status --note --project-id")
                return 1
            runs.update(
                args.run_id,
                status=cast(RunStatus | None, args.status),
                note=args.note,
                project_id=args.project_id,
                updated_at=now_jst_iso(),
            )
            print(f"updated run {args.run_id}")
            return 0

        if args.command == "delete-run":
            if runs.find_by_id(args.run_id) is None:
                print(f"run id {args.run_id} not found")
                return 1
            runs.delete(args.run_id)
            if args.with_files:
                artifact_store.delete_run_dir(args.run_id)
            print(f"deleted run {args.run_id}")
            return 0

        if args.command == "add-project":
            if projects.find(args.project_id) is not None:
                print(f"project id {args.project_id} already exists")
                return 1
            projects.add(
                project_id=args.project_id,
                project_path=args.project_path,
                created_at=now_jst_iso(),
                note=args.note,
            )
            print(f"added project {args.project_id}")
            return 0

        if args.command == "update-project":
            if projects.find(args.project_id) is None:
                print(f"project id {args.project_id} not found")
                return 1
            if args.project_path is None and args.note is None:
                print("update-project requires --project-path or --note")
                return 1
            projects.update(args.project_id, project_path=args.project_path, note=args.note)
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
            rows = runs.list_runs(
                from_at=args.from_at,
                to_at=args.to_at,
                status=cast(RunStatus | None, args.status),
                project_id=args.project_id,
            )
            print("run_id\tproject_id\tstatus\tcreated_at")
            for row in rows:
                print(f"{row.run_id}\t{row.project_id or ''}\t{row.status}\t{row.created_at}")
            return 0

        if args.command == "list-projects":
            rows = projects.list_projects()
            print("project_id\tproject_path\tcreated_at\tnote")
            for row in rows:
                print(f"{row.project_id}\t{row.project_path}\t{row.created_at}\t{row.note}")
            return 0

        return 1
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
