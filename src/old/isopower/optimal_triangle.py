from __future__ import annotations
import numpy as np
from isopower.calc_a import aCalculator
from scipy.spatial import ConvexHull
from rule.gain_matrix import Pool


# 型ヒント
from rule.character import Character, MatchupVector

class OptimalTriangleFinder:
    """
    最適な3点を探索
    """
    def __init__(self, pool: Pool):
        self.points = np.array(pool.get_pxy_list())
        self.characters = pool.get_characters()
        self.result = []

    def find(self):
        """
        最適な3点を探索するアルゴリズム
        """
        hull = ConvexHull(self.points)
        for simplex, eq in zip(hull.simplices, hull.equations):
            normal = eq[:3]  # 法線ベクトル (A, B, C)
            if normal[0] > 0:  # C > 0 なら z 軸方向に上向きの面
                # simplex は 3点のインデックス
                tri_char=[self.characters[index] for index in simplex ]
                calculator = aCalculator(*tri_char)
                a=calculator.calc()
                if calculator.is_inner:
                    self.result.append([a,*tri_char])
    
    def get_result(self) -> list[list]:
        """
        Docstring for get_result
        
        :param self: Description
        :return: 最適な3点の取得。複数該当する可能性に備えて配列で返す
        :rtype: list[list[Any]]

        結果全体の取得
        """

        return self.result

    def get_a(self) -> MatchupVector:
        """
        平行移動ベクトルaの取得。普通一つなのでrusult[0]に決め打ち
        Docstring for get_a
        
        :param self: Description
        :return: Description
        :rtype: Any
        """
        return self.result[0][0]

    def get_optimal_3characters(self) -> set[Character]:
        return {self.result[0][1],self.result[0][2],self.result[0][3]}

    def display(self):
        for element in self.result:
            for e in element:
                print(e)
            print()
