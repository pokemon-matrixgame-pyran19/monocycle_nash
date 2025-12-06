from .character import Character


class GainMatrix:
    def __init__(self, characters: list[Character]):
        self.characters = characters
        self._initialize_matrix()

    def _initialize_matrix(self):
        size = len(self.characters)
        self.matrix = [[0 for _ in range(size)] for _ in range(size)]

    def set_gain(self):

        for i, char_i in enumerate(self.characters):
            for j, char_j in enumerate(self.characters):
                self.matrix[i][j] = self._calculate_gain(char_i, char_j)

    def _calculate_gain(self, char_i: Character, char_j: Character) -> float:
        dot_product = sum(a * b for a, b in zip(char_i.v, char_j.v))
        gain = char_i.p - char_j.p +(char_i.v * char_j.v) * dot_product
        return gain