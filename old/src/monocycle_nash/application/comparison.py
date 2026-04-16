"""Composite UC: 比較分析 - 二つの行列を比較する。"""

from __future__ import annotations

import numpy as np

from monocycle_nash.domain.matrix.base import PayoffMatrix
from monocycle_nash.domain.matrix.approximation import (
    ApproximationQualityEvaluator,
    MaxElementDifferenceDistance,
    PayoffMatrixApproximation,
    PayoffMatrixDistance,
)

from .dto import AnalysisResultDTO, ComparisonResultDTO
from .solve_equilibrium import SolveEquilibriumUseCase


class ComparisonUseCase:
    """比較分析ユースケース。均衡計算は注入された SolveEquilibriumUseCase に委譲する。"""

    def __init__(self, equilibrium_uc: SolveEquilibriumUseCase) -> None:
        self._equilibrium_uc = equilibrium_uc

    def compare(
        self,
        source: PayoffMatrix,
        reference: PayoffMatrix,
    ) -> ComparisonResultDTO:
        """二つの行列を比較分析する。"""
        source_result = self._analyze(source)
        reference_result = self._analyze(reference)

        max_element_distance = self._compute_max_element_distance(source, reference)
        equilibrium_distance = self._compute_equilibrium_distance(
            source_result, reference_result,
        )

        return ComparisonResultDTO(
            source_analysis=source_result,
            reference_analysis=reference_result,
            max_element_distance=max_element_distance,
            equilibrium_distance=equilibrium_distance,
        )

    def compare_with_approximation(
        self,
        source: PayoffMatrix,
        reference: PayoffMatrix,
        approximation: PayoffMatrixApproximation,
        distance: PayoffMatrixDistance,
    ) -> ComparisonResultDTO:
        """近似器と距離指標を使って二つの行列を比較する。"""
        source_result = self._analyze(source)
        reference_result = self._analyze(reference)

        evaluator = ApproximationQualityEvaluator(approximation, distance)
        approx_result = evaluator.evaluate(source, reference)
        quality = (
            approx_result.diagnostics.evaluation.quality
            if approx_result.diagnostics.evaluation.quality is not None
            else None
        )

        max_element_distance = self._compute_max_element_distance(source, reference)

        return ComparisonResultDTO(
            source_analysis=source_result,
            reference_analysis=reference_result,
            max_element_distance=max_element_distance,
            approximation_quality=quality,
        )

    def _analyze(self, matrix: PayoffMatrix) -> AnalysisResultDTO:
        equilibrium = self._equilibrium_uc.execute(matrix)
        return AnalysisResultDTO(matrix=matrix, equilibrium=equilibrium)

    @staticmethod
    def _compute_max_element_distance(
        source: PayoffMatrix, reference: PayoffMatrix,
    ) -> float | None:
        if source.matrix.shape != reference.matrix.shape:
            return None
        return float(np.max(np.abs(source.matrix - reference.matrix)))

    @staticmethod
    def _compute_equilibrium_distance(
        source_result: AnalysisResultDTO,
        reference_result: AnalysisResultDTO,
    ) -> float | None:
        if source_result.equilibrium is None or reference_result.equilibrium is None:
            return None
        source_probs = source_result.equilibrium.mixed_strategy.probabilities
        ref_probs = reference_result.equilibrium.mixed_strategy.probabilities
        if source_probs.shape != ref_probs.shape:
            return None
        return float(np.max(np.abs(source_probs - ref_probs)))

