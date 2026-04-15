"""Mini UC: 単体分析 - Analyze a single PayoffMatrix."""

from __future__ import annotations

import numpy as np

from monocycle_nash.domain.matrix.base import PayoffMatrix
from monocycle_nash.domain.matrix.monocycle import MonocyclePayoffMatrix
from monocycle_nash.domain.solver.selector import SolverSelector

from .dto import AnalysisConfig, AnalysisResultDTO, EquilibriumResultDTO
from .ports import VisualizationPort


class SingleAnalysisUseCase:
    """単体分析ミニユースケース"""

    def __init__(self, visualization: VisualizationPort | None = None):
        self._visualization = visualization

    def analyze(
        self, matrix: PayoffMatrix, config: AnalysisConfig,
    ) -> AnalysisResultDTO:
        """行列に対して指定された分析を実行する。"""
        equilibrium_result: EquilibriumResultDTO | None = None
        payoff_graph_svg: str | None = None
        character_plot_svg: str | None = None

        if config.solve_equilibrium:
            equilibrium_result = self._solve_equilibrium(matrix)

        if config.generate_payoff_graph and self._visualization is not None:
            payoff_graph_svg = self._visualization.draw_payoff_graph(
                matrix=matrix.matrix,
                labels=matrix.labels,
                threshold=config.graph_threshold,
                canvas_size=config.graph_canvas_size,
            )

        if (
            config.generate_character_plot
            and self._visualization is not None
            and isinstance(matrix, MonocyclePayoffMatrix)
        ):
            character_plot_svg = self._visualization.draw_character_plot(
                characters=matrix.characters,
                canvas_size=config.character_canvas_size,
                margin=config.character_margin,
            )

        return AnalysisResultDTO(
            matrix=matrix,
            equilibrium=equilibrium_result,
            payoff_graph_svg=payoff_graph_svg,
            character_plot_svg=character_plot_svg,
        )

    @staticmethod
    def _solve_equilibrium(matrix: PayoffMatrix) -> EquilibriumResultDTO:
        """均衡解を計算し、純粋戦略期待利得・乖離・固有値を求める。"""
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
