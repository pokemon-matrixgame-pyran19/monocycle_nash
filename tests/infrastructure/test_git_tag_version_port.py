from __future__ import annotations

import subprocess
from typing import Any

import pytest

from monocycle_nash import __version__
from monocycle_nash.infrastructure.output import GitTagVersionPort


def test_get_version_returns_first_tag_from_head(monkeypatch: pytest.MonkeyPatch) -> None:
    port = GitTagVersionPort()

    class _CompletedWithTags:
        returncode = 0
        stdout = "v2.0.0\nv1.9.0\n"

    def _fake_run(*args: Any, **kwargs: Any) -> _CompletedWithTags:
        return _CompletedWithTags()

    monkeypatch.setattr(subprocess, "run", _fake_run)
    assert port.get_version() == "v2.0.0"


def test_get_version_falls_back_to_package_version_when_no_tag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    port = GitTagVersionPort()

    class _CompletedWithNoTags:
        returncode = 0
        stdout = "\n"

    def _fake_run(*args: Any, **kwargs: Any) -> _CompletedWithNoTags:
        return _CompletedWithNoTags()

    monkeypatch.setattr(subprocess, "run", _fake_run)
    assert port.get_version() == __version__


def test_get_version_falls_back_to_package_version_on_git_nonzero_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    port = GitTagVersionPort()

    class _CompletedWithError:
        returncode = 128
        stdout = ""

    def _fake_run(*args: Any, **kwargs: Any) -> _CompletedWithError:
        return _CompletedWithError()

    monkeypatch.setattr(subprocess, "run", _fake_run)
    assert port.get_version() == __version__


def test_get_version_falls_back_to_package_version_when_git_not_found(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    port = GitTagVersionPort()

    def _fake_run(*args: Any, **kwargs: Any) -> Any:
        raise FileNotFoundError("git")

    monkeypatch.setattr(subprocess, "run", _fake_run)
    assert port.get_version() == __version__
