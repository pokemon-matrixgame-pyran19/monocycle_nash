import numpy as np
from typing import TYPE_CHECKING

from .base import PayoffMatrix

if TYPE_CHECKING:
    from equilibrium.domain import MixedStrategy


class GeneralPayoffMatrix(PayoffMatrix):
    """
    一般の利得行列
    - 任意の行列要素を持つ
    - nashpyによる線形最適化で均衡解を計算
    """
    
    def __init__(self, matrix: np.ndarray, labels: list[str] | None = None):
        self._matrix = np.array(matrix, dtype=float)
        self._size = matrix.shape[0]
        self._labels = labels or [f"s{i}" for i in range(self._size)]
    
    @property
    def matrix(self) -> np.ndarray:
        return self._matrix
    
    @property
    def size(self) -> int:
        return self._size
    
    @property
    def labels(self) -> list[str]:
        return self._labels
    
    def solve_equilibrium(self) -> "MixedStrategy":
        """nashpyによる線形最適化で均衡解を計算"""
        from solver.selector import SolverSelector
        selector = SolverSelector()
        return selector.solve(self)
