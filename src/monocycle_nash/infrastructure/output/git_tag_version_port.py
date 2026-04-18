"""git タグから実行バージョンを取得する VersionPort 実装。"""

from __future__ import annotations

import subprocess
from pathlib import Path

from monocycle_nash import __version__
from monocycle_nash.domain.experiment_run import VersionPort


class GitTagVersionPort(VersionPort):
    """現在コミットに付与された git タグをバージョン文字列として返す。"""

    def __init__(self, *, cwd: Path | str | None = None) -> None:
        self._cwd = Path(cwd) if cwd is not None else None

    def get_version(self) -> str:
        """HEAD に付与されたタグの先頭を返し、無い場合はパッケージ版を返す。"""
        try:
            result = subprocess.run(
                ["git", "tag", "--points-at", "HEAD", "--sort=-v:refname"],
                cwd=self._cwd,
                check=False,
                capture_output=True,
                text=True,
            )
        except FileNotFoundError:
            return __version__

        if result.returncode != 0:
            return __version__

        tags = [
            stripped
            for line in result.stdout.splitlines()
            if (stripped := line.strip())
        ]
        if not tags:
            return __version__
        return tags[0]
