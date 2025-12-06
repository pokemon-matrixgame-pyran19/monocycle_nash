

class MatchupVector():
    # 二次元ベクトル
    def __init__(self, x,y):
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
