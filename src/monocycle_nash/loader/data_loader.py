"""
用途別の入力データローダー。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .toml_tree import TomlTreeLoader


class ExperimentDataLoader:
    """`data/<class>/<id>/data.toml` 形式の実験条件を読み込む。"""

    def __init__(
        self,
        base_dir: Path | str = "data",
        entry_file: str = "data.toml",
        tree_loader: TomlTreeLoader | None = None,
    ):
        self._base_dir = Path(base_dir)
        self._entry_file = entry_file
        self._tree_loader = tree_loader or TomlTreeLoader()

    def load(self, class_name: str, identifier: str) -> dict[str, Any]:
        target = self._base_dir / class_name / identifier / self._entry_file
        return self._tree_loader.load(target)

    def load_from_path(self, relative_path: Path | str) -> dict[str, Any]:
        path = Path(relative_path)
        if not path.is_absolute():
            path = self._base_dir / path
        if path.suffix:
            return self._tree_loader.load(path)
        return self._tree_loader.load(path / self._entry_file)


class SettingDataLoader:
    """`data/setting` 配下の実行設定を読み込む。"""

    def __init__(
        self,
        base_dir: Path | str = "data/setting",
        tree_loader: TomlTreeLoader | None = None,
    ):
        self._base_dir = Path(base_dir)
        self._tree_loader = tree_loader or TomlTreeLoader()

    def load(self, name: str) -> dict[str, Any]:
        candidate = Path(name)
        if not candidate.suffix:
            candidate = candidate.with_suffix(".toml")
        if not candidate.is_absolute():
            candidate = self._base_dir / candidate
        return self._tree_loader.load(candidate)

    def load_file(self, relative_file: Path | str) -> dict[str, Any]:
        path = Path(relative_file)
        if not path.is_absolute():
            path = self._base_dir / path
        return self._tree_loader.load(path)
