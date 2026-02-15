"""
キャラクターのドメインモデル

MatchupVector: 2次元相性ベクトル
Character: 単相性モデル用キャラクター（power, vector, label）
"""

from __future__ import annotations
import numpy as np
from typing import Literal


class MatchupVector:
    """
    相性ベクトル（2次元ベクトル）- 値オブジェクト
    
    2次元空間でのベクトル演算を提供します。
    外積計算（times）が単相性モデルの核心です。
    """
    
    def __init__(self, x: float | np.ndarray | list | MatchupVector, y: float | None = None):
        """
        Args:
            x: x座標、または[x, y]の配列、またはMatchupVector
            y: y座標（xが配列の場合は不要）
        """
        if isinstance(x, MatchupVector):
            # 既にMatchupVectorならコピー
            self._data = x._data.copy()
            return
        
        if y is None:
            arr = np.asarray(x, dtype=float)
        else:
            arr = np.array([x, y], dtype=float)
        
        if arr.shape != (2,):
            raise ValueError(f"2次元ベクトルである必要があります。shape: {arr.shape}")
        
        self._data = arr
    
    @property
    def x(self) -> float:
        """x座標"""
        return self._data[0]
    
    @property
    def y(self) -> float:
        """y座標"""
        return self._data[1]
    
    def times(self, other: MatchupVector) -> float:
        """
        2次元外積計算（スカラー積）
        
        self × other = self.x * other.y - self.y * other.x
        
        単相性モデルでは、この外積が相性関係を表します。
        """
        return self.x * other.y - self.y * other.x
    
    # 演算子オーバーロード
    def __add__(self, other: MatchupVector) -> MatchupVector:
        """ベクトル加算"""
        return MatchupVector(self._data + other._data)
    
    def __sub__(self, other: MatchupVector) -> MatchupVector:
        """ベクトル減算"""
        return MatchupVector(self._data - other._data)
    
    def __mul__(self, scalar: float) -> MatchupVector:
        """スカラー乗算"""
        return MatchupVector(self._data * scalar)
    
    def __rmul__(self, scalar: float) -> MatchupVector:
        """スカラー乗算（右側）"""
        return self.__mul__(scalar)
    
    def __neg__(self) -> MatchupVector:
        """ベクトル否定（-v）"""
        return MatchupVector(-self._data)
    
    def __truediv__(self, scalar: float) -> MatchupVector:
        """スカラー除算"""
        return MatchupVector(self._data / scalar)
    
    def __repr__(self) -> str:
        return f"MatchupVector({self.x}, {self.y})"
    
    def __str__(self) -> str:
        return f"{self.x},{self.y}"
    
    def __eq__(self, other: MatchupVector | list[float] | np.ndarray) -> bool:
        """
        ベクトルの等価判定
        
        numpyのallcloseを使用して浮動小数点の誤差を許容します。
        """
        if isinstance(other, MatchupVector):
            return np.allclose(self._data, other._data)
        elif isinstance(other, list):
            return np.allclose(self._data, np.array(other))
        elif isinstance(other, np.ndarray):
            return np.allclose(self._data, other)
        else:
            raise TypeError(f"MatchupVectorと比較できない型: {type(other)}")
    
    def copy(self) -> MatchupVector:
        """ベクトルのコピーを作成"""
        return MatchupVector(self._data.copy())
    
    def to_array(self) -> np.ndarray:
        """numpy配列として返す"""
        return self._data.copy()


class Character:
    """
    単相性モデル用キャラクター
    
    Attributes:
        p: パワー値（power）
        v: 相性ベクトル（MatchupVector）
        label: 表示用ラベル
    
    単相性モデルでは、利得行列の要素は以下で計算されます：
        Aij = pi - pj + vi × vj
    """
    
    def __init__(self, power: float, vector: MatchupVector, label: str = ""):
        """
        Args:
            power: パワー値
            vector: 相性ベクトル（MatchupVector）
            label: 表示用ラベル（例: "ピカチュウ"）。省略時は空文字列。
        """
        self.p = float(power)
        self.v = MatchupVector(vector)
        self.label = label
    
    def tolist(self, order: list[Literal["p", "x", "y"]] = []) -> list[float]:
        """
        キャラクターの属性をリストとして取得
        
        Args:
            order: 取得する属性の順序 ["p", "x", "y"] の組み合わせ
                   空リストの場合は [p, x, y] の順
        
        Returns:
            指定された順序の属性値リスト
        """
        if not order:
            return [float(self.p), float(self.v.x), float(self.v.y)]
        
        result = []
        for key in order:
            if key not in ["p", "x", "y"]:
                raise ValueError(f"無効なキー: {key}. 'p', 'x', 'y' のいずれかを指定してください")
            elif key == "p":
                result.append(float(self.p))
            elif key == "x":
                result.append(float(self.v.x))
            elif key == "y":
                result.append(float(self.v.y))
        return result
    
    def convert(self, action_vector: list[float] | MatchupVector) -> Character:
        """
        等パワー座標 a に基づいて新しいキャラクターを生成
        
        単相性モデルでの原点移動：
            p' = p + v × a
            v' = v - a
        
        Args:
            action_vector: 等パワー座標 a（MatchupVectorまたは[x, y]のリスト）
        
        Returns:
            原点移動後の新しいCharacter
        """
        a = MatchupVector(action_vector)
        new_power = self.p + self.v.times(a)
        new_vector = MatchupVector(self.v.x - a.x, self.v.y - a.y)
        return Character(new_power, new_vector, self.label)
    
    def __str__(self) -> str:
        return f"Character(power={self.p}, vector={self.v}, label={self.label!r})"
    
    def __repr__(self) -> str:
        if self.label:
            return f"Character({self.p}, {self.v}, {self.label!r})"
        return f"Character({self.p}, {self.v!r})"
    
    def __eq__(self, other: object) -> bool:
        """
        キャラクターの等価判定
        
        powerとvector、labelの全てが等しい場合に等価とみなします。
        """
        if not isinstance(other, Character):
            return False
        return (abs(self.p - other.p) < 1e-9 and 
                self.v == other.v and
                self.label == other.label)


def get_characters(data: list[list[float]] | np.ndarray, labels: list[str] | None = None) -> list[Character]:
    """
    [p, x, y]の形式のデータからCharacterリストを生成
    
    Args:
        data: [[p1, x1, y1], [p2, x2, y2], ...] の形式のデータ
        labels: 各キャラクターのラベルリスト（省略時は連番）
    
    Returns:
        Characterオブジェクトのリスト
    
    Example:
        >>> data = [[1.0, 1.0, 0.0], [0.5, -0.5, 0.5]]
        >>> chars = get_characters(data, ["ピカチュウ", "カメックス"])
    """
    if labels is None:
        labels = [f"c{i}" for i in range(len(data))]
    return [Character(d[0], MatchupVector(d[1], d[2]), labels[i]) for i, d in enumerate(data)]
