from .character import Character
import numpy as np


class Environment:
    def __init__(self, characters: list[Character]):
        self.characters = characters
        self.set_matrix()

    def set_matrix(self):
        size = len(self.characters)
        temp_matrix = [[0 for _ in range(size)] for _ in range(size)]
        for i, char_i in enumerate(self.characters):
            for j, char_j in enumerate(self.characters):
                temp_matrix[i][j] = self._calculate_gain(char_i, char_j)
        self.matrix = np.array(temp_matrix)

    def get_matrix(self) -> np.array:
        return self.matrix

    def _calculate_gain(self, char_i: Character, char_j: Character) -> float:
        gain = char_i.p - char_j.p +char_i.v.times(char_j.v) 
        return gain