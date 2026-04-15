"""Composite UC: 直接分析 - Build matrix(es) and analyze each."""

from __future__ import annotations

from .dto import AnalysisConfig, AnalysisResultDTO, MatrixInputDTO
from .matrix_construction import MatrixConstructionUseCase
from .ports import OutputPort, RunContext
from .single_analysis import SingleAnalysisUseCase


class DirectAnalysisUseCase:
    """直接分析ユースケース = 行列構築 + 単体分析"""

    def __init__(
        self,
        matrix_builder: MatrixConstructionUseCase,
        analyzer: SingleAnalysisUseCase,
        output: OutputPort,
    ):
        self._matrix_builder = matrix_builder
        self._analyzer = analyzer
        self._output = output

    def execute(
        self, input_dto: MatrixInputDTO, config: AnalysisConfig,
    ) -> list[AnalysisResultDTO]:
        """行列を構築し、全ての関連行列について分析を実行する。"""
        main_matrix, intermediate_matrices = self._matrix_builder.build(input_dto)

        ctx = self._output.create_run_context("direct_analysis")
        self._output.save_input_snapshot(
            ctx, "matrix_input.json", self._serialize_input(input_dto),
        )

        results: list[AnalysisResultDTO] = []

        # Analyze intermediate matrices first (e.g., character matrix before team matrix)
        for idx, inter_matrix in enumerate(intermediate_matrices):
            result = self._analyzer.analyze(inter_matrix, config)
            results.append(result)
            self._write_analysis_result(ctx, result, prefix=f"intermediate_{idx}")

        # Analyze main matrix
        main_result = self._analyzer.analyze(main_matrix, config)
        results.append(main_result)
        self._write_analysis_result(ctx, main_result, prefix="main")

        self._output.write_metadata(ctx, {
            "use_case": "direct_analysis",
            "matrix_count": len(results),
        })

        return results

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

    @staticmethod
    def _serialize_input(input_dto: MatrixInputDTO) -> dict:
        data: dict = {}
        if input_dto.raw_matrix is not None:
            data["raw_matrix"] = input_dto.raw_matrix
        if input_dto.labels is not None:
            data["labels"] = input_dto.labels
        if input_dto.characters is not None:
            data["characters"] = input_dto.characters
        if input_dto.team_mode is not None:
            data["team_mode"] = input_dto.team_mode
        if input_dto.teams is not None:
            data["teams"] = input_dto.teams
        return data
