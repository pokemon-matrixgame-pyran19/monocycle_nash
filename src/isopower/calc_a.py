import numpy as np
from rule.character import MatchupVector

class aCalculator:
    def __init__(self, character1, character2, character3):
        self.c1 = character1
        self.c2 = character2
        self.c3 = character3

    def calc(self) -> MatchupVector:
        """
        等パワーになるようなaを計算する
        a = p1 (v2-v3) + p2 (v3-v1) + p3 (v1-v2) /T
        ここでT = det(u1,u2,u3) , u= (vy, -vx ,p)
        """
        n = self.c1.p*(self.c2.v - self.c3.v) + self.c2.p*(self.c3.v - self.c1.v) + self.c3.p*(self.c1.v - self.c2.v)
        T= (self.c2.v-self.c1.v).times(self.c3.v - self.c1.v)
        a = -n/T
        return a

class DevCalculator(aCalculator):
    def dev_calc(self) -> list[float]:
        """
        calcと同じ計算だが開発用に途中式含めて元の数式に忠実に計算する
        """
        return [0]