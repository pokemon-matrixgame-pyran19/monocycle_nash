from __future__ import annotations

import subprocess
from typing import Any, NoReturn

import pytest

from monocycle_nash import __version__
from monocycle_nash.infrastructure.output import GitTagVersionPort


def test_get_version_returns_first_tag_from_head(monkeypatch: pytest.MonkeyPatch) -> None:
    port = GitTagVersionPort()

    class _DummyCompleted:
        stdout = "v2.0.0\nv1.9.0\n"

    def _fake_run(*args: Any, **kwargs: Any) -> _DummyCompleted:
        return _DummyCompleted()

    monkeypatch.setattr(subprocess, "run", _fake_run)
    assert port.get_version() == "v2.0.0"


def test_get_version_falls_back_to_package_version_when_no_tag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    port = GitTagVersionPort()

    class _DummyCompleted:
        stdout = "\n"

    def _fake_run(*args: Any, **kwargs: Any) -> _DummyCompleted:
        return _DummyCompleted()

    monkeypatch.setattr(subprocess, "run", _fake_run)
    assert port.get_version() == __version__


def test_get_version_falls_back_to_package_version_on_git_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    port = GitTagVersionPort()

    def _fake_run(*args: Any, **kwargs: Any) -> NoReturn:
        raise subprocess.CalledProcessError(128, "git")

    monkeypatch.setattr(subprocess, "run", _fake_run)
    assert port.get_version() == __version__


def test_get_version_falls_back_to_package_version_when_git_not_found(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    port = GitTagVersionPort()

    def _fake_run(*args: Any, **kwargs: Any) -> NoReturn:
        raise FileNotFoundError("git")

    monkeypatch.setattr(subprocess, "run", _fake_run)
    assert port.get_version() == __version__
