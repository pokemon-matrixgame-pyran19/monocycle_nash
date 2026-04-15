from __future__ import annotations

import pytest

import monocycle_nash.main as main_mod
from monocycle_nash.presentation import cli as cli_mod


def test_main_delegates_to_cli_main(monkeypatch: pytest.MonkeyPatch) -> None:
    """main.main() は cli_main() に委譲する。"""
    called = []

    def _fake_cli_main(argv: list[str] | None = None) -> int:
        called.append(argv)
        return 0

    monkeypatch.setattr(cli_mod, "main", _fake_cli_main)
    monkeypatch.setattr(main_mod, "cli_main", _fake_cli_main)

    assert main_mod.main() == 0
    assert len(called) == 1


def test_cli_version_flag(capsys: pytest.CaptureFixture[str]) -> None:
    """--version フラグでバージョンを出力して終了する。"""
    with pytest.raises(SystemExit, match="0"):
        cli_mod.main(["--version"])
    captured = capsys.readouterr()
    assert "0.2.0" in captured.out


def test_cli_missing_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """存在しない設定ファイルを指定するとエラーコード1を返す。"""
    result = cli_mod.main(["--config", "/nonexistent/path.toml"])
    assert result == 1


def test_cli_unsupported_feature(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: pytest.TempPathFactory,
) -> None:
    """未対応の feature 名はエラーコード1を返す。"""
    from pathlib import Path

    fake_config = tmp_path / "fake.toml"  # type: ignore[operator]
    fake_config.write_text("")

    def _fake_load(path: object) -> dict:
        return {"features": ["unknown_feature_xyz"], "shared": {}}

    monkeypatch.setattr(cli_mod, "_load_run_config", _fake_load)
    monkeypatch.setattr(cli_mod, "_DEFAULT_CONFIG", fake_config)

    result = cli_mod.main([])
    assert result == 1
    captured = capsys.readouterr()
    assert "未対応" in captured.err
