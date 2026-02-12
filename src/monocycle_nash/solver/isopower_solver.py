import numpy as np

from .base import EquilibriumSolver
from matrix.monocycle import MonocyclePayoffMatrix
from equilibrium.domain import MixedStrategy
from isopower.calc_a import aCalculator
from isopower.optimal_triangle import OptimalTriangleFinder
from rule.gain_matrix import Pool


class IsopowerSolver(EquilibriumSolver):
    """
    等パワー座標による高速ソルバー
    - 単相性モデル専用
    - 等パワー座標を使った高速解法
    """
    
    def can_solve(self, matrix) -> bool:
        """MonocyclePayoffMatrixのみ対応"""
        return isinstance(matrix, MonocyclePayoffMatrix)
    
    def solve(self, matrix: MonocyclePayoffMatrix) -> MixedStrategy:
        """
        等パワー座標による高速解法で均衡解を計算
        
        アルゴリズム:
        1. 最適な等パワー座標aを探索
        2. aで原点移動
        3. 移動後の空間で均衡を計算
        """
        # Pool経由でOptimalTriangleFinderを使用
        pool = Pool(matrix.characters)
        
        # 最適三角形を探索
        finder = OptimalTriangleFinder(pool)
        finder.find()
        
        result = finder.get_result()
        
        if not result:
            # 有効な三角形がない場合はフォールバック
            return self._fallback_solve(matrix)
        
        # 最初の結果を使用
        a_vector = finder.get_a()
        indices = self._get_character_indices(finder, matrix)
        
        # 等パワー座標で原点移動
        shifted = matrix.shift_origin(a_vector)
        
        # 移動後の均衡を計算
        return self._calculate_shifted_equilibrium(shifted, indices)
    
    def _fallback_solve(self, matrix: MonocyclePayoffMatrix) -> MixedStrategy:
        """フォールバック: 均等分布を返す"""
        probs = np.ones(matrix.size) / matrix.size
        return MixedStrategy(probs, matrix.labels)
    
    def _get_character_indices(
        self, 
        finder: OptimalTriangleFinder,
        matrix: MonocyclePayoffMatrix
    ) -> tuple[int, int, int]:
        """最適3キャラクターのインデックスを取得"""
        optimal_chars = finder.get_optimal_3characters()
        indices = []
        for char in optimal_chars:
            for i, c in enumerate(matrix.characters):
                if c is char or (c.p == char.p and c.v == char.v):
                    indices.append(i)
                    break
        
        # 3つのインデックスを返す
        if len(indices) >= 3:
            return (indices[0], indices[1], indices[2])
        else:
            # フォールバック: 最初の3つ
            return (0, 1, 2)
    
    def _calculate_shifted_equilibrium(
        self, 
        shifted: MonocyclePayoffMatrix, 
        indices: tuple[int, int, int]
    ) -> MixedStrategy:
        """
        原点移動後の均衡を計算
        
        理論: i,j,kの組み合わせに対し、混合戦略の比率は
        i:j:k = Ajk : Aki : Aij
        
        ここで Aij は移動後の利得行列の要素
        """
        i, j, k = indices
        A = shifted.matrix
        
        # 比率を計算: i:j:k = A[j,k] : A[k,i] : A[i,j]
        ratio_i = A[j, k]
        ratio_j = A[k, i]
        ratio_k = A[i, j]
        
        # 確率に正規化（合計が1になるように）
        total = ratio_i + ratio_j + ratio_k
        if abs(total) < 1e-10:
            # 合計が0に近い場合は均等分布
            return self._fallback_solve(shifted)
        
        prob_i = ratio_i / total
        prob_j = ratio_j / total
        prob_k = ratio_k / total
        
        # 全戦略の確率配列を作成（ijk以外は0）
        n = shifted.size
        probs = np.zeros(n)
        probs[i] = prob_i
        probs[j] = prob_j
        probs[k] = prob_k
        
        return MixedStrategy(probs, shifted.labels)
