

class MatchupVector():
    # 二次元ベクトル
    def __init__(self, x:float,y:float):
        self.x = x
        self.y = y

    def times(self, other: 'MatchupVector') -> float:
        """
        ２次元用の外積計算
        """
        return self.x * other.y - self.y * other.x

class Character:
    def __init__(self, power:float, vector: MatchupVector):
        self.p = power
        self.v = vector

    def tolist(self) -> list[float]:
        return [self.p, self.v.x, self.v.y]

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