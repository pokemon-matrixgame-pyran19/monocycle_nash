"""ファイルシステムから行列データを読み込む MatrixDataPort 実装。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from monocycle_nash.application.ports import MatrixDataPort
from monocycle_nash.infrastructure.input.toml_loader import TomlTreeLoader


class FileMatrixDataReader(MatrixDataPort):
    """ファイルシステムから行列データを読み込む。"""

    def __init__(
        self,
        data_dir: Path | str = "data",
        entry_file: str = "data.toml",
        tree_loader: TomlTreeLoader | None = None,
    ) -> None:
        self._data_dir = Path(data_dir)
        self._entry_file = entry_file
        self._loader = tree_loader or TomlTreeLoader()

    def load_matrix_data(self, identifier: str) -> dict[str, Any]:
        """data/matrix/<identifier>/data.toml を読み込む。"""
        path = self._data_dir / "matrix" / identifier / self._entry_file
        return self._loader.load(path)
