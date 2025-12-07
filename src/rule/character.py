import numpy as np
from typing import Union

class MatchupVector():
    # 二次元ベクトル
    def __init__(self, x: Union[float, np.ndarray], y: float|None = None):
        """
        Args:
            x: x座標、または[x, y]の配列
            y: y座標（xが配列の場合は不要）
        """
        if y is None:
            arr = np.asarray(x, dtype=float)
        else:
            arr = np.array([x, y], dtype=float)
        
        if arr.shape != (2,):
            raise ValueError(f"2次元ベクトルである必要があります。shape: {arr.shape}")
        
        self._data = arr
    
    @property
    def x(self) -> float:
        return self._data[0]
    
    @property
    def y(self) -> float:
        return self._data[1]

    def times(self, other: 'MatchupVector') -> float:
        """
        ２次元用の外積計算
        """
        return self.x * other.y - self.y * other.x

# 演算子オーバーロード
    def __add__(self, other: 'MatchupVector') -> 'MatchupVector':
        return MatchupVector(self._data + other._data)
    
    def __sub__(self, other: 'MatchupVector') -> 'MatchupVector':
        return MatchupVector(self._data - other._data)
    
    def __mul__(self, scalar: float) -> 'MatchupVector':
        """スカラー倍"""
        return MatchupVector(self._data * scalar)
    
    def __rmul__(self, scalar: float) -> 'MatchupVector':
        return self.__mul__(scalar)

    def __neg__(self) -> 'MatchupVector':
        """-v"""
        return MatchupVector(-self._data)
    
    def __truediv__(self, scalar: float) -> 'MatchupVector':
        return MatchupVector(self._data / scalar)
    
    def __repr__(self) -> str:
        return f"MatchupVector({self.x}, {self.y})"
    
    def __eq__(self, other: 'MatchupVector') -> bool:
        return np.allclose(self._data, other._data)

class Character:
    def __init__(self, power:float, vector: MatchupVector):
        self.p = power
        self.v = vector

    def tolist(self) -> list[float]:
        return [float(self.p), float(self.v.x), float(self.v.y)]

    def convert(self, action_vector: list[float]) -> "Character":
        """
        aベクトルに基づいて新しいキャラクターを生成
        p' = p + v x a
        v' = v - a
        """
        a = action_vector
        new_power = self.p + self.v.times(MatchupVector(*a))
        new_vector = MatchupVector(self.v.x - a[0], self.v.y- a[1])
        return Character(new_power, new_vector)