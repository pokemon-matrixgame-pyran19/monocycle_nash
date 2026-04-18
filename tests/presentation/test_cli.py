from __future__ import annotations

from pathlib import Path

import pytest

from monocycle_nash.presentation.cli import main


def test_cli_main_runs_and_stores_snapshot(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
method = "general_from_raw"
name = "raw"

[params]
matrix = [[0.0, 1.0], [-1.0, 0.0]]
labels = ["A", "B"]
""".strip(),
        encoding="utf-8",
    )

    result_dir = tmp_path / "result"
    code = main([str(config_path), "--result-dir", str(result_dir)])

    assert code == 0
    snapshot_path = result_dir / "1" / "input" / "config_tree.toml"
    assert snapshot_path.exists()


def test_cli_main_returns_error_for_missing_config(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    code = main(["missing", "--data-dir", str(tmp_path)])

    assert code == 1
    captured = capsys.readouterr()
    assert "実行に失敗しました (FileNotFoundError)" in captured.err
