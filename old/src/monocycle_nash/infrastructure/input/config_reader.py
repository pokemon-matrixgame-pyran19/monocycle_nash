"""ファイルシステムから実験データ・グラフ設定を読み込む Port 実装。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from monocycle_nash.application.ports import ExperimentDataPort, GraphConfigPort
from monocycle_nash.infrastructure.input.toml_loader import TomlTreeLoader


class FileExperimentDataReader(ExperimentDataPort):
    """ファイルシステムから実験データを読み込む。"""

    def __init__(
        self,
        data_dir: Path | str = "data",
        entry_file: str = "data.toml",
        tree_loader: TomlTreeLoader | None = None,
    ) -> None:
        self._data_dir = Path(data_dir)
        self._entry_file = entry_file
        self._loader = tree_loader or TomlTreeLoader()

    def load_experiment_data(self, class_name: str, identifier: str) -> dict[str, Any]:
        """data/<class_name>/<identifier>/data.toml を読み込む。"""
        path = self._data_dir / class_name / identifier / self._entry_file
        return self._loader.load(path)


class FileGraphConfigReader(GraphConfigPort):
    """ファイルシステムからグラフ設定を読み込む。"""

    def __init__(
        self,
        data_dir: Path | str = "data",
        entry_file: str = "data.toml",
        tree_loader: TomlTreeLoader | None = None,
    ) -> None:
        self._data_dir = Path(data_dir)
        self._entry_file = entry_file
        self._loader = tree_loader or TomlTreeLoader()

    def load_graph_config(self, identifier: str) -> dict[str, Any]:
        """data/graph/<identifier>/data.toml を読み込む。"""
        path = self._data_dir / "graph" / identifier / self._entry_file
        return self._loader.load(path)
