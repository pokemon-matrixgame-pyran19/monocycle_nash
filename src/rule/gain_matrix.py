from .character import Character
import numpy as np


class Environment:
    def __init__(self, characters: list[Character]):
        self.characters = characters
        self.size = len(self.characters)
        self.set_matrix()
        self.precision = 1e-4


    def set_matrix(self):
        temp_matrix = [[0 for _ in range(self.size)] for _ in range(self.size)]
        for i, char_i in enumerate(self.characters):
            for j, char_j in enumerate(self.characters):
                temp_matrix[i][j] = self._calculate_gain(char_i, char_j)
        self.matrix = np.array(temp_matrix)

    def get_matrix(self) -> np.array:
        return self.matrix

    def _calculate_gain(self, char_i: Character, char_j: Character) -> float:
        gain = char_i.p - char_j.p +char_i.v.times(char_j.v) 
        return gain

    def __eq__(self, other: 'Environment') -> bool:
        """
        利得行列の一致で等価判定
        """
        return np.allclose(self.matrix, other.matrix, rtol=self.precision, atol=self.precision)

    def convert(self, action_vector: list[float]) -> "Environment":
        """
        aベクトルに基づいて新しい環境を生成
        """
        new_characters = [c.convert(action_vector) for c in self.characters]
        return Environment(new_characters)

class BatchEnvironment(Environment):
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
        #np_action_matrix = np.tile(np.hstack((0,np.array(np_action_vector))),(self.size,1))

        dv = np.tile(np.array(np_action_vector),(self.size,1))
        cross_a = np.vstack((np_action_vector[1],-np_action_vector[0]))
        dp = self.characters[:,1:3].dot(cross_a)
        # cross_a = np.hstack((np_action_vector[1],-np_action_vector[0]))
        # dp = cross_a.dot(self.characters[:,1:3].T)

        new_characters=self.characters + np.hstack((dp.reshape(-1,1),-dv))

        return BatchEnvironment(new_characters)