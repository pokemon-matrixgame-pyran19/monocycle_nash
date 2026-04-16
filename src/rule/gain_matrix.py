from __future__ import annotations
import numpy as np

# 型ヒント
from .character import Character, MatchupVector
from typing import Literal

class Pool:
    def __init__(self, characters: list[Character]):
        self.characters = characters
        self.size = len(self.characters)
        self.set_matrix()
        self.precision = 1e-4

    def set_matrix(self):
        temp_matrix:list[list[float]] = [[0 for _ in range(self.size)] for _ in range(self.size)]
        for i, char_i in enumerate(self.characters):
            for j, char_j in enumerate(self.characters):
                temp_matrix[i][j] = self._calculate_gain(char_i, char_j)
        self.matrix = np.array(temp_matrix)

    def get_matrix(self) -> np.ndarray:
        return self.matrix

    def _calculate_gain(self, char_i: Character, char_j: Character) -> float:
        gain = char_i.p - char_j.p +char_i.v.times(char_j.v) 
        return gain

    def __eq__(self, other: 'Pool') -> bool:
        """
        利得行列の一致で等価判定
        """
        return np.allclose(self.matrix, other.matrix, rtol=self.precision, atol=self.precision)

    def convert(self, action_vector: list[float]|MatchupVector) -> "Pool":
        """
        aベクトルに基づいて新しい環境を生成
        平行移動なんだから関数名shiftとかな気がしてきた。後で関数名変更
        """
        new_characters = [c.convert(action_vector) for c in self.characters]
        return Pool(new_characters)

    def get_pxy_list(self, order:list[Literal["p","x","y"]]=[]) -> list[list[float]]:
        """
        環境パラメータを二次元リストで取得
        """
        if order == []:
            return [c.tolist() for c in self.characters]
        else:
            return [c.tolist(order) for c in self.characters]

    def get_characters(self):
        return self.characters

class BatchEnvironment(Pool):
    """
     一括処理用に最適化したアルゴリズム使うもの 
     データをnumpyで保持。
     一行ごとにcharacterの関数呼び出すオーバーヘッド避けて、こちらのクラス中で外積計算等を実装
    """
    def __init__(self, characters: list[Character] | np.ndarray):
        if isinstance(characters, np.ndarray):
            self.characters = characters
            self.size = len(self.characters)
        elif isinstance(characters, list):
            self.characters = np.array([ c.tolist() for c in characters])
            self.size = len(self.characters)
        else:
            raise TypeError("characters must be list or np.ndarray")
        self.set_matrix()
        self.precision = 1e-4

    def set_matrix(self):
        """
        行列生成

        charactersが次の形の二次元numpyになってるの注意
        [
            (power, vx, vy),
            (power, vx, vy),
            (power, vx, vy),
            (power, vx, vy)
        ]
        """
        power_vector = np.tile(self.characters[:,0],(self.size,1))
        P = power_vector.T - power_vector

        matrix1=self.characters[:,1:3]
        matrix2=np.vstack((self.characters[:,2],-self.characters[:,1]))

        V = matrix1.dot(matrix2)

        self.matrix = P + V


    def convert(self, action_vector: list[float]) -> "BatchEnvironment":
        np_action_vector = np.array(action_vector)

        dv = np.tile(np.array(np_action_vector),(self.size,1))
        cross_a = np.vstack((np_action_vector[1],-np_action_vector[0]))
        dp = self.characters[:,1:3].dot(cross_a)

        new_characters=self.characters + np.hstack((dp.reshape(-1,1),-dv))

        return BatchEnvironment(new_characters)