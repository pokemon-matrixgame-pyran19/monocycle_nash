"""Helpers for creating per-project references to run output directories."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess

from .clock import now_jst_iso


def create_analysis_project_reference(*, run_id: int, result_dir: Path, project_path: str | None, status: str) -> None:
    if not project_path:
        return

    refs_dir = Path(project_path) / "experiment_refs"
    refs_dir.mkdir(parents=True, exist_ok=True)

    ref_dir = refs_dir / str(run_id)
    _remove_existing_reference(ref_dir)

    target_dir = result_dir.resolve()
    try:
        ref_dir.symlink_to(target_dir, target_is_directory=True)
        _remove_existing_reference(refs_dir / f"{run_id}.txt")
        return
    except OSError:
        pass

    if _try_create_windows_junction(link_dir=ref_dir, target_dir=target_dir):
        _remove_existing_reference(refs_dir / f"{run_id}.txt")
        return

    (refs_dir / f"{run_id}.txt").write_text(
        "\n".join(
            [
                f"result_path={target_dir}",
                f"created_at={now_jst_iso()}",
                f"status={status}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _remove_existing_reference(path: Path) -> None:
    if not path.exists() and not path.is_symlink():
        return
    if path.is_symlink() or path.is_file():
        path.unlink()
        return
    shutil.rmtree(path)


def _try_create_windows_junction(*, link_dir: Path, target_dir: Path) -> bool:
    if os.name != "nt":
        return False
    try:
        subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link_dir), str(target_dir)],
            check=True,
            capture_output=True,
            text=True,
        )
        return True
    except (subprocess.CalledProcessError, OSError):
        return False

