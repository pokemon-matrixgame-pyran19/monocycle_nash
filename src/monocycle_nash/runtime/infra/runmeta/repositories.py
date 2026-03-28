"""Repository layer for projects/runs persistence."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from .models import ProjectRecord, RunRecord, RunStatus

UNASSIGNED_PROJECT_ID = "_null"


@dataclass
class RunsRepository:
    conn: sqlite3.Connection

    def create_running(
        self,
        *,
        command: str,
        git_commit: str | None,
        note: str,
        project_id: str | None,
        started_at: str,
        created_at: str,
        updated_at: str,
    ) -> int:
        cur = self.conn.execute(
            """
            INSERT INTO runs(created_at, started_at, ended_at, status, command, git_commit, note, project_id, updated_at)
            VALUES(?, ?, NULL, 'running', ?, ?, ?, ?, ?)
            """,
            (created_at, started_at, command, git_commit, note, project_id, updated_at),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def finish(self, *, run_id: int, status: RunStatus, ended_at: str, updated_at: str) -> None:
        self.conn.execute(
            """
            UPDATE runs
            SET status = ?, ended_at = ?, updated_at = ?
            WHERE run_id = ?
            """,
            (status, ended_at, updated_at, run_id),
        )
        self.conn.commit()

    def update(
        self,
        run_id: int,
        *,
        status: RunStatus | None = None,
        note: str | None = None,
        project_id: str | None = None,
        updated_at: str,
    ) -> None:
        fields: list[str] = []
        values: list[object] = []
        if status is not None:
            fields.append("status = ?")
            values.append(status)
        if note is not None:
            fields.append("note = ?")
            values.append(note)
        if project_id is not None:
            fields.append("project_id = ?")
            values.append(project_id)
        fields.append("updated_at = ?")
        values.append(updated_at)
        values.append(run_id)
        self.conn.execute(f"UPDATE runs SET {', '.join(fields)} WHERE run_id = ?", tuple(values))
        self.conn.commit()

    def delete(self, run_id: int) -> None:
        self.conn.execute("DELETE FROM runs WHERE run_id = ?", (run_id,))
        self.conn.commit()

    def find_by_id(self, run_id: int) -> RunRecord | None:
        row = self.conn.execute(
            """
            SELECT runs.*, projects.project_path AS project_path
            FROM runs
            LEFT JOIN projects ON projects.project_id = runs.project_id
            WHERE runs.run_id = ?
            """,
            (run_id,),
        ).fetchone()
        if row is None:
            return None
        return RunRecord(**dict(row))

    def list_runs(
        self,
        *,
        from_at: str | None = None,
        to_at: str | None = None,
        status: RunStatus | None = None,
        project_id: str | None = None,
    ) -> list[RunRecord]:
        query = """
            SELECT runs.*, projects.project_path AS project_path
            FROM runs
            LEFT JOIN projects ON projects.project_id = runs.project_id
        """
        clauses: list[str] = []
        params: list[object] = []
        if from_at is not None:
            clauses.append("runs.created_at >= ?")
            params.append(from_at)
        if to_at is not None:
            clauses.append("runs.created_at <= ?")
            params.append(to_at)
        if status is not None:
            clauses.append("runs.status = ?")
            params.append(status)
        if project_id is not None:
            if project_id == UNASSIGNED_PROJECT_ID:
                clauses.append("runs.project_id IS NULL")
            else:
                clauses.append("runs.project_id = ?")
                params.append(project_id)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY runs.created_at DESC"
        rows = self.conn.execute(query, tuple(params)).fetchall()
        return [RunRecord(**dict(r)) for r in rows]


@dataclass
class ProjectsRepository:
    conn: sqlite3.Connection

    def add(self, *, project_id: str, project_path: str, created_at: str, note: str = "") -> None:
        self.conn.execute(
            "INSERT INTO projects(project_id, project_path, created_at, note) VALUES(?, ?, ?, ?)",
            (project_id, project_path, created_at, note),
        )
        self.conn.commit()

    def update(self, project_id: str, *, project_path: str | None = None, note: str | None = None) -> None:
        fields: list[str] = []
        values: list[object] = []
        if project_path is not None:
            fields.append("project_path = ?")
            values.append(project_path)
        if note is not None:
            fields.append("note = ?")
            values.append(note)
        if not fields:
            return
        values.append(project_id)
        self.conn.execute(f"UPDATE projects SET {', '.join(fields)} WHERE project_id = ?", tuple(values))
        self.conn.commit()

    def delete(self, project_id: str) -> None:
        self.conn.execute("DELETE FROM projects WHERE project_id = ?", (project_id,))
        self.conn.commit()

    def find(self, project_id: str) -> ProjectRecord | None:
        row = self.conn.execute("SELECT * FROM projects WHERE project_id = ?", (project_id,)).fetchone()
        if row is None:
            return None
        return ProjectRecord(**dict(row))

    def list_projects(self) -> list[ProjectRecord]:
        rows = self.conn.execute("SELECT * FROM projects ORDER BY created_at DESC").fetchall()
        return [ProjectRecord(**dict(r)) for r in rows]
