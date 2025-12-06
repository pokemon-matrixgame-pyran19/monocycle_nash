from .character import Character
import numpy as np


class Environment:
    def __init__(self, characters: list[Character]):
        self.characters = characters
        self.set_matrix()
        self.precision = 1e-4

        self.size = len(self.characters)

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
        行動ベクトルに基づいて新しい環境を生成
        """
        new_characters = []
        for i, char in enumerate(self.characters):
            a = action_vector[i]
            new_power = char.p + char.v.times(Character(0, MatchupVector(a,0)).v)
            new_vector = MatchupVector(char.v.x - a, char.v.y)
            new_characters.append(Character(new_power, new_vector))
        return Environment(new_characters)