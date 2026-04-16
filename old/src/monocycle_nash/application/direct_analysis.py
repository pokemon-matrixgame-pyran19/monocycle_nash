"""Composite UC: 直接分析 - 行列を構築し、注入された各分析ユースケースを実行する。"""

from __future__ import annotations

from monocycle_nash.domain.matrix.monocycle import MonocyclePayoffMatrix

from .draw_character_plot import DrawCharacterPlotUseCase
from .draw_payoff_graph import DrawPayoffGraphUseCase
from .dto import AnalysisResultDTO
from .matrix_build import MatrixBuildUseCase
from .ports import OutputPort, RunContext
from .solve_equilibrium import SolveEquilibriumUseCase


class DirectAnalysisUseCase:
    """
    直接分析ユースケース。

    注入された分析ユースケース（SolveEquilibrium, DrawPayoffGraph, DrawCharacterPlot）の
    うち None でないものを実行する。どの分析を行うかは依存性注入によって決定し、
    このクラス自身はフラグで分岐しない。
    """

    def __init__(
        self,
        matrix_build_uc: MatrixBuildUseCase,
        output: OutputPort,
        equilibrium_uc: SolveEquilibriumUseCase | None = None,
        payoff_graph_uc: DrawPayoffGraphUseCase | None = None,
        character_plot_uc: DrawCharacterPlotUseCase | None = None,
    ) -> None:
        self._matrix_build_uc = matrix_build_uc
        self._output = output
        self._equilibrium_uc = equilibrium_uc
        self._payoff_graph_uc = payoff_graph_uc
        self._character_plot_uc = character_plot_uc

    def execute(
        self,
        matrix_id: str,
        graph_id: str | None = None,
    ) -> list[AnalysisResultDTO]:
        """
        行列を構築し、注入された全分析ユースケースを実行する。

        Args:
            matrix_id: MatrixDataPort に渡す行列識別子
            graph_id: グラフ生成系ユースケースに渡すグラフ設定識別子

        Returns:
            各行列（中間行列 + メイン行列）の分析結果リスト
        """
        main_matrix, intermediate_matrices = self._matrix_build_uc.execute(matrix_id)

        ctx = self._output.create_run_context("direct_analysis")
        self._output.save_input_snapshot(
            ctx, "matrix_input.json", {"matrix_id": matrix_id},
        )

        results: list[AnalysisResultDTO] = []

        for idx, inter_matrix in enumerate(intermediate_matrices):
            result = self._run_analysis(inter_matrix, graph_id)
            results.append(result)
            self._write_analysis_result(ctx, result, prefix=f"intermediate_{idx}")

        main_result = self._run_analysis(main_matrix, graph_id)
        results.append(main_result)
        self._write_analysis_result(ctx, main_result, prefix="main")

        self._output.write_metadata(ctx, {
            "use_case": "direct_analysis",
            "matrix_count": len(results),
        })

        return results

    def _run_analysis(
        self,
        matrix: object,
        graph_id: str | None,
    ) -> AnalysisResultDTO:
        from monocycle_nash.domain.matrix.base import PayoffMatrix
        assert isinstance(matrix, PayoffMatrix)

        equilibrium = (
            self._equilibrium_uc.execute(matrix) if self._equilibrium_uc else None
        )

        payoff_graph_svg: str | None = None
        if self._payoff_graph_uc is not None and graph_id is not None:
            payoff_graph_svg = self._payoff_graph_uc.execute(matrix, graph_id)

        character_plot_svg: str | None = None
        if (
            self._character_plot_uc is not None
            and graph_id is not None
            and isinstance(matrix, MonocyclePayoffMatrix)
        ):
            character_plot_svg = self._character_plot_uc.execute(matrix, graph_id)

        return AnalysisResultDTO(
            matrix=matrix,
            equilibrium=equilibrium,
            payoff_graph_svg=payoff_graph_svg,
            character_plot_svg=character_plot_svg,
        )

    def _write_analysis_result(
        self,
        ctx: RunContext,
        result: AnalysisResultDTO,
        prefix: str,
    ) -> None:
        if result.equilibrium is not None:
            eq = result.equilibrium
            self._output.write_json(ctx, f"{prefix}_equilibrium.json", {
                "strategy_ids": eq.mixed_strategy.strategy_ids,
                "probabilities": eq.mixed_strategy.probabilities.tolist(),
            })
            self._output.write_json(ctx, f"{prefix}_pure_strategy.json", {
                "strategy_ids": result.matrix.row_strategies.ids,
                "labels": result.matrix.labels,
                "payoffs": eq.pure_payoffs,
            })
            self._output.write_json(ctx, f"{prefix}_divergence.json", {
                "strategy_ids": result.matrix.row_strategies.ids,
                "divergence": eq.divergence,
            })
            if eq.eigenvalues is not None:
                self._output.write_json(ctx, f"{prefix}_eigenvalues.json", {
                    "eigenvalues": eq.eigenvalues,
                })

        if result.payoff_graph_svg is not None:
            self._output.write_svg(ctx, f"{prefix}_edge_graph.svg", result.payoff_graph_svg)

        if result.character_plot_svg is not None:
            self._output.write_svg(
                ctx, f"{prefix}_character_vector.svg", result.character_plot_svg,
            )

