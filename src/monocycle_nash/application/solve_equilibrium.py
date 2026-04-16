"""Mini UC: 利得行列の均衡解を計算する。"""

from __future__ import annotations

import numpy as np

from monocycle_nash.domain.matrix.base import PayoffMatrix
from monocycle_nash.domain.solver.selector import SolverSelector

from .dto import EquilibriumResultDTO


class SolveEquilibriumUseCase:
    """ユースケース: 利得行列に対して均衡解を計算し、純粋戦略期待利得・乖離・固有値を求める。"""

    def execute(self, matrix: PayoffMatrix) -> EquilibriumResultDTO:
        """均衡解を計算する。"""
        selector = SolverSelector()
        eq = selector.solve(matrix)

        pure_payoff = (
            np.asarray(matrix.matrix, dtype=float)
            @ np.asarray(eq.probabilities, dtype=float)
        )
        pure_payoffs = pure_payoff.tolist()

        best = float(np.max(pure_payoff)) if pure_payoff.size else 0.0
        divergence = [float(best - x) for x in pure_payoffs]

        eigenvalues: list[float] | None = None
        if matrix.is_alternating():
            eigenvalues = matrix.eigenvalues().tolist()

        return EquilibriumResultDTO(
            mixed_strategy=eq,
            pure_payoffs=pure_payoffs,
            divergence=divergence,
            eigenvalues=eigenvalues,
        )
