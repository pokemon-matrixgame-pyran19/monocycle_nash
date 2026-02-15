"""SQLite connection and migration for run metadata."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SQLiteConnectionFactory:
    db_path: str | Path

    def connect(self) -> sqlite3.Connection:
        db_path = Path(self.db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn


def migrate(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS projects (
          project_id TEXT PRIMARY KEY,
          project_path TEXT NOT NULL,
          created_at TEXT NOT NULL,
          note TEXT DEFAULT ''
        );

        CREATE TABLE IF NOT EXISTS runs (
          run_id INTEGER PRIMARY KEY AUTOINCREMENT,
          created_at TEXT NOT NULL,
          started_at TEXT NOT NULL,
          ended_at TEXT,
          status TEXT NOT NULL CHECK(status IN ('running','success','fail','killed')),
          command TEXT NOT NULL,
          git_commit TEXT,
          note TEXT DEFAULT '',
          project_id TEXT,
          updated_at TEXT NOT NULL,
          FOREIGN KEY(project_id) REFERENCES projects(project_id)
            ON UPDATE CASCADE ON DELETE SET NULL
        );

        CREATE INDEX IF NOT EXISTS idx_runs_created_at ON runs(created_at);
        CREATE INDEX IF NOT EXISTS idx_runs_status ON runs(status);
        CREATE INDEX IF NOT EXISTS idx_runs_project_id ON runs(project_id);
        """
    )
    conn.commit()
