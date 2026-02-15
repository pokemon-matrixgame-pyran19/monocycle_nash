"""Repository layer for run metadata records backed by SQLite."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from .clock import now_jst_iso
from .models import RunStatus


def _as_dict(row: sqlite3.Row | None) -> dict[str, object] | None:
    if row is None:
        return None
    return dict(row)


@dataclass
class ProjectsRepository:
    """Access and manage project records."""

    conn: sqlite3.Connection

    def add(self, name: str) -> int:
        """Create a project and return its id."""

        timestamp = now_jst_iso()
        cursor = self.conn.execute(
            """
            INSERT INTO projects(name, created_at, updated_at)
            VALUES(?, ?, ?)
            """,
            (name, timestamp, timestamp),
        )
        self.conn.commit()
        return int(cursor.lastrowid)

    def find(self, project_id: int) -> dict[str, object] | None:
        """Fetch a project by id."""

        row = self.conn.execute(
            "SELECT id, name, created_at, updated_at FROM projects WHERE id = ?",
            (project_id,),
        ).fetchone()
        return _as_dict(row)

    def update(self, project_id: int, *, name: str) -> None:
        """Update project fields."""

        self.conn.execute(
            """
            UPDATE projects
            SET name = ?, updated_at = ?
            WHERE id = ?
            """,
            (name, now_jst_iso(), project_id),
        )
        self.conn.commit()

    def delete(self, project_id: int) -> None:
        """Delete a project."""

        self.conn.execute("DELETE FROM projects WHERE id = ?", (project_id,))
        self.conn.commit()

    # Compatibility methods for older service wiring.
    def save(self, project_id: int, name: str) -> dict[str, object] | None:
        timestamp = now_jst_iso()
        self.conn.execute(
            """
            INSERT INTO projects(id, name, created_at, updated_at)
            VALUES(?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                name = excluded.name,
                updated_at = excluded.updated_at
            """,
            (project_id, name, timestamp, timestamp),
        )
        self.conn.commit()
        return self.find(project_id)

    def get(self, project_id: int) -> dict[str, object] | None:
        return self.find(project_id)


@dataclass
class RunsRepository:
    """Access and manage run records."""

    conn: sqlite3.Connection

    def create_running(self, project_id: int) -> int:
        """Create a running run and return last inserted run id."""

        timestamp = now_jst_iso()
        cursor = self.conn.execute(
            """
            INSERT INTO runs(project_id, status, created_at, ended_at, updated_at)
            VALUES(?, 'running', ?, NULL, ?)
            """,
            (project_id, timestamp, timestamp),
        )
        self.conn.commit()
        return int(cursor.lastrowid)

    def finish(self, run_id: int, status: RunStatus) -> None:
        """Set terminal status, ended_at and updated_at for a run."""

        timestamp = now_jst_iso()
        self.conn.execute(
            """
            UPDATE runs
            SET status = ?, ended_at = ?, updated_at = ?
            WHERE id = ?
            """,
            (status, timestamp, timestamp, run_id),
        )
        self.conn.commit()

    def update(
        self,
        run_id: int,
        *,
        status: RunStatus | None = None,
        ended_at: str | None = None,
        project_id: int | None = None,
    ) -> None:
        """Update mutable run fields and always refresh updated_at."""

        fields: list[str] = []
        values: list[object] = []

        if status is not None:
            fields.append("status = ?")
            values.append(status)
        if ended_at is not None:
            fields.append("ended_at = ?")
            values.append(ended_at)
        if project_id is not None:
            fields.append("project_id = ?")
            values.append(project_id)

        fields.append("updated_at = ?")
        values.append(now_jst_iso())
        values.append(run_id)

        self.conn.execute(
            f"UPDATE runs SET {', '.join(fields)} WHERE id = ?",
            tuple(values),
        )
        self.conn.commit()

    def delete(self, run_id: int) -> None:
        """Delete a run by id."""

        self.conn.execute("DELETE FROM runs WHERE id = ?", (run_id,))
        self.conn.commit()

    def find_by_id(self, run_id: int) -> dict[str, object] | None:
        """Fetch a run by id."""

        row = self.conn.execute(
            """
            SELECT id, project_id, status, created_at, ended_at, updated_at
            FROM runs
            WHERE id = ?
            """,
            (run_id,),
        ).fetchone()
        return _as_dict(row)


    # Compatibility methods for older service wiring.
    def create(self, run_id: int, project_id: int) -> dict[str, object] | None:
        timestamp = now_jst_iso()
        self.conn.execute(
            """
            INSERT INTO runs(id, project_id, status, created_at, ended_at, updated_at)
            VALUES(?, ?, 'running', ?, NULL, ?)
            """,
            (run_id, project_id, timestamp, timestamp),
        )
        self.conn.commit()
        return self.find_by_id(run_id)

    def update_status(self, run_id: int, status: RunStatus) -> dict[str, object] | None:
        self.finish(run_id=run_id, status=status)
        return self.find_by_id(run_id)

    def get(self, run_id: int) -> dict[str, object] | None:
        return self.find_by_id(run_id)

    def list_runs(
        self,
        *,
        from_at: str | None = None,
        to_at: str | None = None,
        status: RunStatus | None = None,
        project_id: int | None = None,
    ) -> list[dict[str, object]]:
        """List runs with optional dynamic filters."""

        query = "SELECT id, project_id, status, created_at, ended_at, updated_at FROM runs"
        clauses: list[str] = []
        params: list[object] = []

        if from_at is not None:
            clauses.append("created_at >= ?")
            params.append(from_at)
        if to_at is not None:
            clauses.append("created_at <= ?")
            params.append(to_at)
        if status is not None:
            clauses.append("status = ?")
            params.append(status)
        if project_id is not None:
            clauses.append("project_id = ?")
            params.append(project_id)

        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY created_at DESC, id DESC"

        rows = self.conn.execute(query, tuple(params)).fetchall()
        return [dict(row) for row in rows]


# Backward-compatible aliases.
ProjectRepository = ProjectsRepository
RunRepository = RunsRepository
