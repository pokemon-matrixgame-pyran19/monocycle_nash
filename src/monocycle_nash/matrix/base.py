from abc import ABC, abstractmethod
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..equilibrium.domain import MixedStrategy
    from ..strategy.domain import PureStrategySet


class PayoffMatrix(ABC):
    """利得行列の抽象基底クラス"""

    @property
    @abstractmethod
    def matrix(self) -> np.ndarray:
        """利得行列のnumpy配列"""
        pass

    @property
    @abstractmethod
    def size(self) -> int:
        """行列のサイズ"""
        pass

    @property
    @abstractmethod
    def row_strategies(self) -> "PureStrategySet":
        """行プレイヤー側の純粋戦略集合"""
        pass

    @property
    @abstractmethod
    def col_strategies(self) -> "PureStrategySet":
        """列プレイヤー側の純粋戦略集合"""
        pass

    @property
    def labels(self) -> list[str]:
        """後方互換: 行プレイヤー側ラベル。"""
        return self.row_strategies.labels

    @abstractmethod
    def solve_equilibrium(self) -> "MixedStrategy":
        """この行列に適した方法で均衡解を計算"""
        pass

    def get_value(self, i: int, j: int) -> float:
        """行列の要素(i,j)を取得"""
        return self.matrix[i, j]

    def is_alternating(self, *, atol: float = 1e-8) -> bool:
        """交代行列( A^T = -A ) かどうかを判定する。"""
        matrix = np.asarray(self.matrix, dtype=float)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            return False
        return np.allclose(matrix + matrix.T, 0.0, atol=atol)

    def eigenvalues(self, *, atol: float = 1e-8) -> np.ndarray:
        """交代行列の固有値から、虚部の絶対値を実数配列で返す。"""
        if not self.is_alternating(atol=atol):
            raise ValueError("eigenvalues は交代行列の場合のみ利用できます")
        values = np.linalg.eigvals(self.matrix)
        return np.abs(np.imag(values))

    def __eq__(self, other: object) -> bool:
        """利得行列の等価判定"""
        if not isinstance(other, PayoffMatrix):
            return False
        return np.allclose(self.matrix, other.matrix, rtol=1e-4, atol=1e-4)
