"""
最適三角形探索

キャラクターの凸包から、等パワー座標aが内部にある最適な三角形を探索します。
"""

import numpy as np
from scipy.spatial import ConvexHull
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .calc_a import aCalculator
from ..character.domain import Character, MatchupVector

if TYPE_CHECKING:
    from ..matrix.monocycle import MonocyclePayoffMatrix


@dataclass
class OptimalTriangleResult:
    """最適三角形探索の結果"""
    a_vector: MatchupVector  # 等パワー座標
    indices: tuple[int, int, int]  # 3点のインデックス
    characters: tuple[Character, Character, Character]  # 3キャラクター


class OptimalTriangleFinder:
    """
    最適な3点（ナッシュ均衡を形成する三角形）を探索
    
    探索アルゴリズム:
    1. キャラクターの凸包（ConvexHull）を計算
    2. 凸包の各面（三角形）に対して:
       a. その3点で等パワー座標 a を計算
       b. a が三角形の内部にあるか判定（is_inner）
    3. 内部にaがある三角形を最適な組み合わせとして返す
    
    理論的根拠:
    - 平行移動前の点 vi, vj, vk の内側に a がある
    - または同じ意味だが平行移動後の vi, vj, vk が原点を囲む
    - この条件を満たすとき、等パワーの3点が最大パワーになりナッシュ均衡となる
    """
    
    def __init__(self, matrix: "MonocyclePayoffMatrix"):
        """
        Args:
            matrix: 単相性モデルの利得行列
        """
        self._matrix = matrix
        self._characters = matrix.characters
        # 凸包計算用の点群 [p, x, y]
        self._points = np.array([
            [c.p, c.v.x, c.v.y] for c in self._characters
        ])
        self._result: OptimalTriangleResult | None = None
    
    def find(self) -> None:
        """
        最適な3点を探索するアルゴリズム
        
        凸包の上面（法線のx成分が正）の三角形から、
        is_innerがTrueのものを探して結果に保存します。
        """
        n = len(self._characters)
        if n < 3:
            return
        
        # 3点の場合は直接計算（ConvexHullは4点以上必要）
        if n == 3:
            self._check_triangle(0, 1, 2)
            return
        
        # 4点以上の場合は凸包を計算
        try:
            hull = ConvexHull(self._points)
            
            for simplex, eq in zip(hull.simplices, hull.equations):
                # simplex: 3点のインデックス
                # eq: 面の方程式 [A, B, C, D] (Ax + By + Cz + D = 0)
                normal = eq[:3]  # 法線ベクトル (A, B, C)
                
                # 上面を選択（法線のx成分が正）
                if normal[0] > 0:
                    i, j, k = simplex[0], simplex[1], simplex[2]
                    if self._check_triangle(i, j, k):
                        return  # 見つかったら終了
        except Exception:
            # ConvexHullが失敗した場合は全ての三角形をチェック
            for i in range(n - 2):
                for j in range(i + 1, n - 1):
                    for k in range(j + 1, n):
                        if self._check_triangle(i, j, k):
                            return
    
    def _check_triangle(self, i: int, j: int, k: int) -> bool:
        """
        指定された3点で等パワー座標aを計算し、is_innerか判定
        
        Args:
            i, j, k: キャラクターのインデックス
            
        Returns:
            is_innerがTrueで結果を保存した場合はTrue、そうでなければFalse
        """
        tri_char = (
            self._characters[i],
            self._characters[j],
            self._characters[k]
        )
        
        # 等パワー座標aを計算
        calculator = aCalculator(*tri_char)
        a = calculator.calc()
        
        # aが三角形の内部にあるか判定
        if calculator.is_inner:
            self._result = OptimalTriangleResult(
                a_vector=a,
                indices=(i, j, k),
                characters=tri_char
            )
            return True
        return False
    
    def find_best(self) -> OptimalTriangleResult | None:
        """
        最適な三角形を探索して返す
        
        Returns:
            最適三角形の結果、見つからない場合はNone
        """
        self.find()
        return self._result
    
    def get_result(self) -> list[OptimalTriangleResult]:
        """
        結果のリストを取得
        
        Returns:
            最適三角形の結果リスト（複数該当する可能性に備えて配列で返す）
        """
        if self._result is None:
            return []
        return [self._result]
    
    def get_a(self) -> MatchupVector:
        """
        等パワー座標aを取得
        
        Returns:
            等パワー座標a
        
        Raises:
            ValueError: 結果が見つからない場合
        """
        if self._result is None:
            raise ValueError("最適三角形が見つかっていません。find()を先に呼び出してください。")
        return self._result.a_vector
    
    def get_optimal_3characters(self) -> set[Character]:
        """
        最適3キャラクターを取得
        
        Returns:
            最適3キャラクターのセット
        
        Raises:
            ValueError: 結果が見つからない場合
        """
        if self._result is None:
            raise ValueError("最適三角形が見つかっていません。find()を先に呼び出してください。")
        return set(self._result.characters)
