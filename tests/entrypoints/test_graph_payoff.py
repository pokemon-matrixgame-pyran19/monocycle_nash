from __future__ import annotations

import sqlite3
from pathlib import Path

from monocycle_nash.entrypoints import graph_payoff


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8")


def _prepare_data_tree(root: Path) -> tuple[Path, Path, Path]:
    data_dir = root / "data"
    db_path = root / ".runmeta" / "run_history.db"
    output_base = root / "result"

    _write(
        data_dir / "run_config" / "baseline" / "rps3_graph.toml",
        '''
        matrix = "rps3"
        graph = "payoff/default"
        setting = "local"
        ''',
    )
    _write(
        data_dir / "matrix" / "rps3" / "data.toml",
        '''
        matrix = [
            [0.0, 1.0, -1.0],
            [-1.0, 0.0, 1.0],
            [1.0, -1.0, 0.0],
        ]
        labels = ["rock", "paper", "scissors"]
        ''',
    )
    _write(
        data_dir / "graph" / "payoff" / "default" / "data.toml",
        '''
        threshold = 0.0
        canvas_size = 700
        ''',
    )
    _write(
        data_dir / "setting" / "local.toml",
        f'''
        [runmeta]
        sqlite_path = "{db_path.as_posix()}"

        [output]
        base_dir = "{output_base.as_posix()}"
        ''',
    )
    return data_dir, db_path, output_base


def test_graph_payoff_main_runs_end_to_end_from_run_config(tmp_path: Path) -> None:
    data_dir, db_path, output_base = _prepare_data_tree(tmp_path)

    code = graph_payoff.main(["--run-config", "baseline/rps3_graph", "--data-dir", str(data_dir)])

    assert code == 0

    conn = sqlite3.connect(db_path)
    try:
        run_id, status = conn.execute("SELECT run_id, status FROM runs ORDER BY run_id DESC LIMIT 1").fetchone()
    finally:
        conn.close()

    run_dir = output_base / str(run_id)
    assert status == "success"
    assert (run_dir / "output" / "edge_graph.svg").exists()
    assert (run_dir / "input" / "matrix.toml").exists()
    assert (run_dir / "input" / "graph.toml").exists()
    assert (run_dir / "input" / "setting.toml").exists()


def test_graph_payoff_records_full_command_with_data_dir(tmp_path: Path) -> None:
    data_dir, db_path, _ = _prepare_data_tree(tmp_path)

    code = graph_payoff.main(["--run-config", "baseline/rps3_graph", "--data-dir", str(data_dir)])

    assert code == 0

    conn = sqlite3.connect(db_path)
    try:
        (command,) = conn.execute("SELECT command FROM runs ORDER BY run_id DESC LIMIT 1").fetchone()
    finally:
        conn.close()

    assert "--run-config baseline/rps3_graph" in command
    assert f"--data-dir {data_dir}" in command
