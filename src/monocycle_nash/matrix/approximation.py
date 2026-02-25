from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

import numpy as np

from .base import PayoffMatrix
from .general import GeneralPayoffMatrix
from .monocycle import MonocyclePayoffMatrix
from ..solver.selector import SolverSelector

InputMatrixT = TypeVar("InputMatrixT", bound=PayoffMatrix)
ApproxMatrixT = TypeVar("ApproxMatrixT", bound=PayoffMatrix)
ReferenceMatrixT = TypeVar("ReferenceMatrixT", bound=PayoffMatrix)


class PayoffMatrixApproximation(ABC, Generic[InputMatrixT, ApproxMatrixT]):
    """利得行列近似の抽象基底クラス。"""

    @abstractmethod
    def approximate(self, matrix: InputMatrixT) -> ApproxMatrixT:
        raise NotImplementedError

    def quality_parameters(self, matrix: InputMatrixT, *, config: dict[str, Any] | None = None) -> dict[str, Any]:
        """精度評価時に付与する補助パラメータを返す。デフォルトは空。"""
        return {}


class MonocycleToGeneralApproximation(PayoffMatrixApproximation[MonocyclePayoffMatrix, GeneralPayoffMatrix]):
    """MonocyclePayoffMatrix を GeneralPayoffMatrix へ変換する近似。"""

    def approximate(self, matrix: MonocyclePayoffMatrix) -> GeneralPayoffMatrix:
        return GeneralPayoffMatrix(
            matrix=np.array(matrix.matrix, dtype=float, copy=True),
            row_strategies=matrix.row_strategies,
            col_strategies=matrix.col_strategies,
        )


class DominantEigenpairMonocycleApproximation(PayoffMatrixApproximation[PayoffMatrix, GeneralPayoffMatrix]):
    """交代行列から絶対値最大の純虚固有値ペアに対応するランク2成分を抽出する。"""

    def __init__(self, *, atol: float = 1e-8):
        self.atol = atol

    def approximate(self, matrix: PayoffMatrix) -> GeneralPayoffMatrix:
        source = np.asarray(matrix.matrix, dtype=float)
        if source.ndim != 2 or source.shape[0] != source.shape[1]:
            raise ValueError("正方行列が必要です")
        if not matrix.is_alternating(atol=self.atol):
            raise ValueError("交代行列( A^T = -A )のみ対応しています")

        approx = self._extract_dominant_component(source)
        return GeneralPayoffMatrix(
            matrix=approx,
            row_strategies=matrix.row_strategies,
            col_strategies=matrix.col_strategies,
        )

    def quality_parameters(self, matrix: PayoffMatrix, *, config: dict[str, Any] | None = None) -> dict[str, Any]:
        source = np.asarray(matrix.matrix, dtype=float)
        ratio = self._dominant_eigen_ratio(source)
        bin_edges = self._resolve_ratio_bin_edges(config)
        return {
            "dominant_eigen_ratio": ratio,
            "dominant_eigen_ratio_bin": self._histogram_label(ratio, bin_edges),
        }

    @staticmethod
    def _extract_dominant_component(source: np.ndarray) -> np.ndarray:
        eigenvalues, eigenvectors = np.linalg.eig(source)
        idx = int(np.argmax(np.abs(np.imag(eigenvalues))))
        sigma = float(np.abs(np.imag(eigenvalues[idx])))
        if sigma < 1e-12:
            return np.zeros_like(source)

        v = eigenvectors[:, idx]
        phase = np.exp(-1j * np.pi / 4.0)
        x = np.real(v * phase)
        y = np.imag(v * phase)

        x_norm = np.linalg.norm(x)
        if x_norm < 1e-12:
            raise ValueError("固有ベクトルの実部が退化しており近似を構成できません")
        q1 = x / x_norm

        y_orth = y - float(np.dot(q1, y)) * q1
        y_norm = np.linalg.norm(y_orth)
        if y_norm < 1e-12:
            raise ValueError("固有ベクトルから独立な2軸を抽出できません")
        q2 = y_orth / y_norm

        sigma_eff = float(abs(q1 @ source @ q2))
        return sigma_eff * (np.outer(q1, q2) - np.outer(q2, q1))

    @staticmethod
    def _dominant_eigen_ratio(source: np.ndarray) -> float:
        eigenvalues = np.linalg.eigvals(source)
        imag_abs = np.abs(np.imag(eigenvalues))
        significant = np.sort(imag_abs[imag_abs >= 1e-12])[::-1]
        if significant.size == 0:
            return float("inf")

        unique_levels: list[float] = []
        for value in significant:
            if not unique_levels or abs(value - unique_levels[-1]) > 1e-9:
                unique_levels.append(float(value))
            if len(unique_levels) >= 2:
                break

        if len(unique_levels) < 2 or unique_levels[1] < 1e-12:
            return float("inf")
        return float(unique_levels[0] / unique_levels[1])

    @staticmethod
    def _resolve_ratio_bin_edges(config: dict[str, Any] | None) -> list[float]:
        raw = None if config is None else config.get("dominant_eigen_ratio_bin_edges")
        if raw is None:
            return [1.25, 1.5, 2.0, 3.0]
        if not isinstance(raw, list) or any(not isinstance(x, (int, float)) for x in raw):
            raise ValueError("dominant_eigen_ratio_bin_edges は数値配列で指定してください")
        edges = [float(x) for x in raw]
        if any(x <= 0.0 for x in edges):
            raise ValueError("dominant_eigen_ratio_bin_edges は正の数値で指定してください")
        if any(edges[i] >= edges[i + 1] for i in range(len(edges) - 1)):
            raise ValueError("dominant_eigen_ratio_bin_edges は昇順で指定してください")
        return edges

    @staticmethod
    def _histogram_label(value: float, edges: list[float]) -> str:
        if not np.isfinite(value):
            return "[inf]"
        lower = 1.0
        for edge in edges:
            if value < edge:
                return f"[{lower:.3f},{edge:.3f})"
            lower = edge
        return f"[{edges[-1]:.3f},inf)"


class PayoffMatrixDistance(ABC, Generic[ApproxMatrixT, ReferenceMatrixT]):
    """利得行列間距離の抽象基底クラス。"""

    @abstractmethod
    def calculate(self, left: ApproxMatrixT, right: ReferenceMatrixT) -> float:
        raise NotImplementedError


class MaxElementDifferenceDistance(PayoffMatrixDistance[PayoffMatrix, PayoffMatrix]):
    """d(A, B) = max_ij |Aij - Bij|。"""

    def calculate(self, left: PayoffMatrix, right: PayoffMatrix) -> float:
        if left.matrix.shape != right.matrix.shape:
            raise ValueError("matrix shapes must match for distance calculation")
        return float(np.max(np.abs(left.matrix - right.matrix)))


class EquilibriumUStrategyDifferenceDistance(PayoffMatrixDistance[PayoffMatrix, PayoffMatrix]):
    """d(A, B) = ||(A-B)u||∞, u は基準行列 B の均衡混合戦略。"""

    def __init__(self, *, solver_selector: SolverSelector | None = None):
        self._solver_selector = solver_selector or SolverSelector()

    def calculate(self, left: PayoffMatrix, right: PayoffMatrix) -> float:
        if left.matrix.shape != right.matrix.shape:
            raise ValueError("matrix shapes must match for distance calculation")

        equilibrium = self._solver_selector.solve(right)
        u = np.asarray(equilibrium.probabilities, dtype=float)

        if u.ndim != 1:
            raise ValueError("equilibrium strategy must be a vector")
        if right.matrix.shape[1] != u.shape[0]:
            raise ValueError("equilibrium strategy size does not match matrix columns")

        diff = (left.matrix - right.matrix) @ u
        return float(np.max(np.abs(diff)))


class ApproximationQualityEvaluator(Generic[InputMatrixT, ApproxMatrixT, ReferenceMatrixT]):
    """近似器と距離指標を組み合わせて近似精度を評価する。"""

    def __init__(
        self,
        approximation: PayoffMatrixApproximation[InputMatrixT, ApproxMatrixT],
        distance: PayoffMatrixDistance[ApproxMatrixT, ReferenceMatrixT],
    ):
        self.approximation = approximation
        self.distance = distance

    def evaluate(self, source: InputMatrixT, reference: ReferenceMatrixT) -> float:
        approximated = self.approximation.approximate(source)
        return self.distance.calculate(approximated, reference)
