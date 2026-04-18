from __future__ import annotations

import os
from pathlib import Path

import pytest

from monocycle_nash.presentation.cli import main


def test_cli_main_runs_and_stores_snapshot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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

    monkeypatch.chdir(tmp_path)
    code = main([str(config_path)])

    assert code == 0
    snapshot_path = tmp_path / "result" / "1" / "input" / "config_tree.toml"
    assert snapshot_path.exists()


def test_cli_main_runs_with_refs_characters(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_path = tmp_path / "config_refs.toml"
    config_path.write_text(
        """
method = "monocycle_from_characters"
name = "ref-root"

[refs]
characters = "characters/rps"
""".strip(),
        encoding="utf-8",
    )

    data_dir = tmp_path / "data" / "characters"
    data_dir.mkdir(parents=True)
    (data_dir / "rps.toml").write_text(
        """
[[characters]]
power = 1.0
vector = [1.0, 0.0]
label = "Rock"

[[characters]]
power = 0.0
vector = [0.0, 1.0]
label = "Paper"
""".strip(),
        encoding="utf-8",
    )

    monkeypatch.chdir(tmp_path)
    code = main([str(config_path), "--data-dir", str(tmp_path / "data")])

    assert code == 0
    snapshot_path = tmp_path / "result" / "1" / "input" / "config_tree.toml"
    assert snapshot_path.exists()


def test_cli_main_returns_error_for_missing_config(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    code = main(["missing", "--data-dir", str(tmp_path)])

    assert code == 1
    captured = capsys.readouterr()
    assert "実行に失敗しました (FileNotFoundError)" in captured.err
