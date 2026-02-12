import numpy as np
import nashpy as nash

from .base import EquilibriumSolver
from ..matrix.general import GeneralPayoffMatrix
from ..equilibrium.domain import MixedStrategy


class NashpySolver(EquilibriumSolver):
    """
    nashpyによる線形最適化ソルバー
    - 一般の利得行列に対応
    - 線形計画法で厳密解を求める
    """
    
    def can_solve(self, matrix) -> bool:
        """全ての行列に適用可能"""
        return True
    
    def solve(self, matrix: GeneralPayoffMatrix) -> MixedStrategy:
        """
        nashpyでナッシュ均衡を計算
        対称ゲームを仮定し、行プレイヤーの均衡戦略を返す
        """
        A = matrix.matrix
        game = nash.Game(A)
        
        # 線形計画法で均衡を計算
        equilibria = list(game.linear_program())
        
        if not equilibria:
            # 均衡が見つからない場合は均等分布を返す
            probs = np.ones(matrix.size) / matrix.size
            return MixedStrategy(probs, matrix.labels)
        
        # 最初の均衡を使用（行プレイヤーの戦略）
        # nashpy.linear_program()はnumpy配列を直接返す
        sigma_r = equilibria[0]
        return MixedStrategy(sigma_r, matrix.labels)
