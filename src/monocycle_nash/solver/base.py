from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..matrix.base import PayoffMatrix
    from ..equilibrium.domain import MixedStrategy


class EquilibriumSolver(ABC):
    """均衡解ソルバーの抽象基底クラス"""
    
    @abstractmethod
    def solve(self, matrix: "PayoffMatrix") -> "MixedStrategy":
        """利得行列から均衡解を計算"""
        pass
    
    @abstractmethod
    def can_solve(self, matrix: "PayoffMatrix") -> bool:
        """このソルバーで解けるか判定"""
        pass
