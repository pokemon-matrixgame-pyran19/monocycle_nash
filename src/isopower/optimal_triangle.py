from rule.character import Character
import numpy as np
from isopower.surface import Surface
from isopower.calc_a import aCalculator
from scipy.spatial import ConvexHull



class OptimalTriangleFinder:
    """
    最適な3点を探索
    """
    def __init__(self, characters: list[Character]):
        self.points = np.array([c.tolist() for c in characters])
        self.characters=characters
        self.result=[]

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

    def display(self):
        for element in self.result:
            for e in element:
                print(e)
            print()
