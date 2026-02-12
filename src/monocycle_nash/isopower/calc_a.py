"""
等パワー座標 a の計算

3つのキャラクターが等パワーになるような座標aを計算します。
"""

from ..character.domain import Character, MatchupVector


class aCalculator:
    """
    等パワー座標 a の計算器
    
    3つのキャラクター c1, c2, c3 に対し、等パワーになるような座標 a を計算します。
    
    理論:
        a = (p1(v2-v3) + p2(v3-v1) + p3(v1-v2)) / T
        
    ここで T は2次元外積（スカラー積）:
        T = (v2-v1) × (v3-v1)
    
    Attributes:
        c1, c2, c3: 3つのキャラクター
        _inner: aが3点の内部にあるか
        _edge: aが3点の辺上にあるか
    """
    
    def __init__(self, character1: Character, character2: Character, character3: Character):
        """
        Args:
            character1: キャラクター1
            character2: キャラクター2
            character3: キャラクター3
        """
        self.c1 = character1
        self.c2 = character2
        self.c3 = character3
        self._inner = False
        self._edge = False
    
    def calc(self) -> MatchupVector:
        """
        等パワー座標aを計算する
        
        Returns:
            等パワー座標a（MatchupVector）
        """
        # 分子: n = p1(v2-v3) + p2(v3-v1) + p3(v1-v2)
        n = (
            self.c1.p * (self.c2.v - self.c3.v) +
            self.c2.p * (self.c3.v - self.c1.v) +
            self.c3.p * (self.c1.v - self.c2.v)
        )
        
        # 分母: T = (v2-v1) × (v3-v1)
        T = (self.c2.v - self.c1.v).times(self.c3.v - self.c1.v)
        
        # a = -n / T
        if abs(T) < 1e-10:
            # Tが0に近い場合（3点が共線）、内部判定をFalseにして原点を返す
            self._inner = False
            self._edge = False
            return MatchupVector(0.0, 0.0)
        
        a = -n / T
        
        # aが三角形の内部にあるか判定
        self._judge_inner(a)
        
        return a
    
    @property
    def is_inner(self) -> bool:
        """
        aが3点の三角形の内部にあるかどうか
        
        Returns:
            内部にあればTrue、外部にあればFalse
        """
        return self._inner
    
    @property
    def is_edge(self) -> bool:
        """
        aが3点の三角形の辺上にあるかどうか
        
        Returns:
            辺上にあればTrue
        """
        return self._edge
    
    def _judge_inner(self, a: MatchupVector):
        """
        aが3点の三角形の内部にあるかどうかを判定
        
        外積の符号で判定します:
        - s1 = (a-c1.v) × (c2.v-c1.v)
        - s2 = (a-c2.v) × (c3.v-c2.v)
        - s3 = (a-c3.v) × (c1.v-c3.v)
        
        すべて同じ符号なら内側、異なる符号があれば外側。
        
        Args:
            a: 判定する座標
        """
        s1 = (a - self.c1.v).times(self.c2.v - self.c1.v)
        s2 = (a - self.c2.v).times(self.c3.v - self.c2.v)
        s3 = (a - self.c3.v).times(self.c1.v - self.c3.v)
        
        # すべて非負またはすべて非正なら内部
        self._inner = (s1 >= 0 and s2 >= 0 and s3 >= 0) or (s1 <= 0 and s2 <= 0 and s3 <= 0)
        
        # いずれかが0なら辺上
        self._edge = abs(s1) < 1e-10 or abs(s2) < 1e-10 or abs(s3) < 1e-10
