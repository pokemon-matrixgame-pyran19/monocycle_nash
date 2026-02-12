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
        self.judge_inner(a)
        return a

    @property
    def is_inner(self) -> bool:
        """
        aが3点の内側にあるかどうか
        """
        return self._inner

    @property
    def is_edge(self) -> bool:
        """
        aが3点の辺上にあるかどうか
        """
        return self._edge

    def judge_inner(self,a: MatchupVector):
        """
        aが3点の内側にあるかどうか
        外積の符号で判定

        1aが12より右側か左側かが1a x 12の符号でわかる
        2aが23より右側か左側かが2a x 23の符号でわかる
        3aが31より右側か左側かが3a x 31の符号でわかる
        すべて同じ符号なら内側、異なる符号があれば外側
        """
        s1=(a- self.c1.v).times(self.c2.v - self.c1.v)
        s2=(a- self.c2.v).times(self.c3.v - self.c2.v)
        s3=(a- self.c3.v).times(self.c1.v - self.c3.v)

        self._inner = (s1>=0 and s2>=0 and s3>=0) or (s1<=0 and s2<=0 and s3<=0)
        self._edge = (s1==0) or (s2==0) or (s3==0)

class DevCalculator(aCalculator):
    def dev_calc(self) -> list[float]:
        """
        calcと同じ計算だが開発用に途中式含めて元の数式に忠実に計算する
        """
        return [0]