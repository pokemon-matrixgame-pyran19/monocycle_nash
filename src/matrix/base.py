from abc import ABC, abstractmethod
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from equilibrium.domain import MixedStrategy

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

    @abstractmethod
    def solve_equilibrium(self) -> "MixedStrategy | None":
        """この行列に適した方法で均衡解を計算"""
        pass

    def get_value(self, i: int, j: int) -> float:
        return float(self.matrix[i, j])
