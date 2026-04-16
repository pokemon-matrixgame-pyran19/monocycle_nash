"""Mini UC: 生行列データから利得行列を構築する。"""

from __future__ import annotations

import numpy as np

from monocycle_nash.domain.matrix.base import PayoffMatrix
from monocycle_nash.domain.matrix.builder import PayoffMatrixBuilder


class BuildMatrixFromRawUseCase:
    """ユースケース: インフラ層が読み込んだ生行列データからドメイン層の利得行列を初期化する。"""

    def execute(self, data: dict) -> PayoffMatrix:
        """
        生行列データから利得行列を構築する。

        Args:
            data: MatrixDataPort が返す dict。"matrix" キーに 2次元リスト必須。

        Returns:
            PayoffMatrix
        """
        self._validate(data)
        matrix = np.asarray(data["matrix"], dtype=float)
        labels: list[str] | None = data.get("labels")
        return PayoffMatrixBuilder.from_general_matrix(matrix, labels=labels)

    @staticmethod
    def _validate(data: dict) -> None:
        raw = data.get("matrix")
        if raw is None:
            raise ValueError("data に 'matrix' キーがありません")
        matrix = np.asarray(raw, dtype=float)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("matrix は正方 2 次元配列である必要があります")
        labels = data.get("labels")
        if labels is not None and len(labels) != matrix.shape[0]:
            raise ValueError("labels 数と matrix サイズが一致しません")
