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
    
    def __eq__(self, other: object) -> bool:
        """利得行列の等価判定"""
        if not isinstance(other, PayoffMatrix):
            return False
        return np.allclose(self.matrix, other.matrix, rtol=1e-4, atol=1e-4)
