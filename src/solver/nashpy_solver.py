import numpy as np
import nashpy as nash
from .base import EquilibriumSolver
from matrix.base import PayoffMatrix
from equilibrium.domain import MixedStrategy

class NashpySolver(EquilibriumSolver):
    """
    nashpyによる線形最適化ソルバー
    - 一般の利得行列に対応
    - 二人零和ゲームとして解く
    - 線形計画法で厳密解を求める
    """

    def can_solve(self, matrix: PayoffMatrix) -> bool:
        """全ての行列に適用可能"""
        return True

    def solve(self, matrix: PayoffMatrix) -> MixedStrategy | None:
        """
        nashpyでナッシュ均衡を計算
        行プレイヤーの均衡戦略を返す
        """
        A = matrix.matrix
        # 二人零和ゲームとして定義
        game = nash.Game(A)

        try:
            # 線形計画法で均衡を計算 (零和ゲーム用)
            # nashpyのlinear_programは [row_strategy, col_strategy] を返す
            equilibria = list(game.linear_program())
        except Exception:
            # 計算に失敗した場合などは None を返す
            return None

        if len(equilibria) < 1:
            return None

        # 行プレイヤーの戦略を取得
        sigma_r = equilibria[0]

        # numpy配列として確実に扱う
        probs = np.array(sigma_r)

        # 行列に保持されているラベルを使用
        labels = getattr(matrix, 'labels', [f"s{i}" for i in range(matrix.size)])

        return MixedStrategy(probs, labels)
