import numpy as np
from .base import PayoffMatrix
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from equilibrium.domain import MixedStrategy
    from solver.nashpy_solver import NashpySolver

class GeneralPayoffMatrix(PayoffMatrix):
    """
    一般の利得行列
    - 任意の行列要素を持つ
    - nashpyによる線形最適化で均衡解を計算
    """

    def __init__(self, matrix: np.ndarray, labels: list[str] | None = None):
        self._matrix = matrix
        self._size = matrix.shape[0]
        self._labels = labels or [f"s{i}" for i in range(self._size)]
        self._solver = None # Will be initialized or imported when needed to avoid circular imports

    @property
    def matrix(self) -> np.ndarray:
        return self._matrix

    @property
    def size(self) -> int:
        return self._size

    @property
    def labels(self) -> list[str]:
        return self._labels

    def solve_equilibrium(self) -> "MixedStrategy | None":
        """nashpyによる線形最適化で均衡解を計算"""
        if self._solver is None:
            from solver.nashpy_solver import NashpySolver
            self._solver = NashpySolver()
        return self._solver.solve(self)
