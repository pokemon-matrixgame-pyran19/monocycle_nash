import numpy as np
from typing import Union, Literal

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

    def __str__(self) -> str:
        return f"{self.x},{self.y}"
    
    def __eq__(self, other: Union['MatchupVector',list[float],np.ndarray]) -> bool:
        if isinstance(other, MatchupVector):
            return np.allclose(self._data, other._data)
        elif isinstance(other,list):
            return np.allclose(self._data, np.array(other))
        elif isinstance(other,np.ndarray):
            return np.allclose(self._data, other)
        else:
            raise TypeError("相性ベクトルが想定しない値と比較されました")

class Character:
    def __init__(self, power:float, vector: MatchupVector):
        self.p = power
        self.v = vector

    def tolist(self,order:list[Literal["p","x","y"]]=[]) -> list[float]:
        if order == []:
            return [float(self.p), float(self.v.x), float(self.v.y)]
        result = []
        for key in order:
            if key not in ["p","x","y"]:
                raise ValueError(f"Invalid key in order: {key}")
            elif key == "p":
                result.append(float(self.p))
            elif key == "x":
                result.append(float(self.v.x))
            elif key == "y":
                result.append(float(self.v.y))
        return result

    def convert(self, action_vector: list[float]|MatchupVector) -> "Character":
        """
        aベクトルに基づいて新しいキャラクターを生成
        p' = p + v x a
        v' = v - a
        """
        if isinstance(action_vector,list):
            a = MatchupVector(*action_vector)
        elif isinstance(action_vector,MatchupVector):
            a = action_vector
        else:
            raise TypeError("想定しない型がaとして入力されました")
        new_power = self.p + self.v.times(a)
        new_vector = MatchupVector(self.v.x - a.x, self.v.y- a.y)
        return Character(new_power, new_vector)

    def __str__(self):
        return f"power:{self.p}, vector: {self.v}"


def get_characters(data:list|np.ndarray) -> list[Character]:
    """
    p,x,yの順で並んだデータを受け取りcharacerのリストを返す
    """
    if isinstance(data,list):
        result = [ Character(d[0],MatchupVector(d[1],d[2])) for d in data ]
        return result
    elif isinstance(data,np.ndarray):
        result = [ Character(d[0],MatchupVector(d[1:])) for d in data ]
        return result
    else:
        raise