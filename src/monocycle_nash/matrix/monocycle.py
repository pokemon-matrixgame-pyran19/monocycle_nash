import numpy as np
from typing import TYPE_CHECKING

from .base import PayoffMatrix
from character.domain import Character, MatchupVector

if TYPE_CHECKING:
    from ..equilibrium.domain import MixedStrategy


class MonocyclePayoffMatrix(PayoffMatrix):
    """
    単相性モデルの利得行列
    - Character(p, v)から生成: Aij = pi - pj + vi×vj
    - 等パワー座標による高速解法を使用可能
    - 元のCharacter情報を保持
    """
    
    def __init__(self, characters: list[Character], labels: list[str] | None = None):
        self._characters = characters
        self._size = len(characters)
        self._labels = labels or [f"c{i}" for i in range(self._size)]
        self._matrix = self._calculate_matrix()
    
    @property
    def matrix(self) -> np.ndarray:
        return self._matrix
    
    @property
    def size(self) -> int:
        return self._size
    
    @property
    def labels(self) -> list[str]:
        return self._labels
    
    @property
    def characters(self) -> list[Character]:
        """元のCharacterリスト（等パワー座標計算に使用）"""
        return self._characters
    
    def _calculate_matrix(self) -> np.ndarray:
        """Aij = pi - pj + vi×vj を計算"""
        n = self._size
        A = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                A[i, j] = (
                    self._characters[i].p - self._characters[j].p +
                    self._characters[i].v.times(self._characters[j].v)
                )
        return A
    
    def solve_equilibrium(self) -> "MixedStrategy":
        """等パワー座標による高速解法で均衡解を計算"""
        from solver.selector import SolverSelector
        selector = SolverSelector()
        return selector.solve(self)
    
    def shift_origin(self, a_vector: MatchupVector) -> "MonocyclePayoffMatrix":
        """
        等パワー座標aで原点移動
        p' = p + v × a
        v' = v - a
        """
        new_characters = [c.convert(a_vector) for c in self._characters]
        return MonocyclePayoffMatrix(new_characters, self._labels)
    
    def get_power_vector(self) -> np.ndarray:
        """パワーベクトルを取得"""
        return np.array([c.p for c in self._characters])
    
    def get_matchup_vectors(self) -> np.ndarray:
        """相性ベクトルを取得 (N×2)"""
        return np.array([[c.v.x, c.v.y] for c in self._characters])
