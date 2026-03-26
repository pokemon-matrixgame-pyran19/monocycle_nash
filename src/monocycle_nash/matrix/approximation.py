from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from .base import PayoffMatrix
from .general import GeneralPayoffMatrix
from ..solver.selector import SolverSelector


class ApproximationMethodDiagnostics(ABC):
    """近似法固有の診断情報の基底。"""


@dataclass(frozen=True)
class EmptyApproximationMethodDiagnostics(ApproximationMethodDiagnostics):
    """近似法固有の診断情報を持たない場合の空オブジェクト。"""


@dataclass(frozen=True)
class DominantEigenpairMethodDiagnostics(ApproximationMethodDiagnostics):
    dominant_eigen_ratio: float


@dataclass(frozen=True)
class ApproximationEvaluation:
    """距離指標などに基づく客観評価。"""

    quality: float | None = None


@dataclass(frozen=True)
class ApproximationDiagnostics:
    """近似器固有の診断情報と評価時に付与される診断情報。"""

    method: ApproximationMethodDiagnostics = field(default_factory=EmptyApproximationMethodDiagnostics)
    evaluation: ApproximationEvaluation = field(default_factory=ApproximationEvaluation)


@dataclass(frozen=True)
class ApproximationResult:
    """近似後の行列と診断情報。"""

    matrix: PayoffMatrix
    diagnostics: ApproximationDiagnostics = field(default_factory=ApproximationDiagnostics)


class PayoffMatrixApproximation(ABC):
    """利得行列近似の抽象基底クラス。"""

    @abstractmethod
    def approximate(self, matrix: PayoffMatrix) -> ApproximationResult:
        raise NotImplementedError


class MonocycleToGeneralApproximation(PayoffMatrixApproximation):
    """MonocyclePayoffMatrix を GeneralPayoffMatrix へ変換する近似。"""

    def approximate(self, matrix: PayoffMatrix) -> ApproximationResult:
        return ApproximationResult(
            matrix=GeneralPayoffMatrix(
                matrix=np.array(matrix.matrix, dtype=float, copy=True),
                row_strategies=matrix.row_strategies,
                col_strategies=matrix.col_strategies,
            )
        )


class DominantEigenpairMonocycleApproximation(PayoffMatrixApproximation):
    """交代行列から絶対値最大の純虚固有値ペアに対応するランク2成分を抽出する。"""

    def __init__(self, *, atol: float = 1e-8):
        self.atol = atol

    def approximate(self, matrix: PayoffMatrix) -> ApproximationResult:
        source = np.asarray(matrix.matrix, dtype=float)
        if source.ndim != 2 or source.shape[0] != source.shape[1]:
            raise ValueError("正方行列が必要です")
        if not matrix.is_alternating(atol=self.atol):
            raise ValueError("交代行列( A^T = -A )のみ対応しています")

        approx = self._extract_dominant_component(source)
        ratio = self._dominant_eigen_ratio(source)
        return ApproximationResult(
            matrix=GeneralPayoffMatrix(
                matrix=approx,
                row_strategies=matrix.row_strategies,
                col_strategies=matrix.col_strategies,
            ),
            diagnostics=ApproximationDiagnostics(
                method=DominantEigenpairMethodDiagnostics(dominant_eigen_ratio=ratio)
            ),
        )

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

class EquilibriumPreservingResidualMonocycleApproximation(PayoffMatrixApproximation):
    """A=J+R を B=J+(p_i-p_j) へ近似し、基準均衡 u に対する作用 Au を保つ。"""

    def __init__(self, *, solver_selector: SolverSelector | None = None, atol: float = 1e-8):
        self._solver_selector = solver_selector or SolverSelector()
        self.atol = atol

    def approximate(self, matrix: PayoffMatrix) -> ApproximationResult:
        source = np.asarray(matrix.matrix, dtype=float)
        if source.ndim != 2 or source.shape[0] != source.shape[1]:
            raise ValueError("正方行列が必要です")
        if not matrix.is_alternating(atol=self.atol):
            raise ValueError("交代行列( A^T = -A )のみ対応しています")

        dominant = DominantEigenpairMonocycleApproximation._extract_dominant_component(source)
        residual = source - dominant

        equilibrium = self._solver_selector.solve(matrix)
        u = np.asarray(equilibrium.probabilities, dtype=float)
        if u.ndim != 1 or u.shape[0] != source.shape[1]:
            raise ValueError("均衡混合戦略の次元が行列サイズと一致しません")

        p = self._solve_potential_vector(u, residual @ u)
        transformed_residual = np.outer(p, np.ones_like(p)) - np.outer(np.ones_like(p), p)
        approx = dominant + transformed_residual

        return ApproximationResult(
            matrix=GeneralPayoffMatrix(
                matrix=approx,
                row_strategies=matrix.row_strategies,
                col_strategies=matrix.col_strategies,
            )
        )

    @staticmethod
    def _solve_potential_vector(u: np.ndarray, rhs: np.ndarray) -> np.ndarray:
        size = u.shape[0]
        operator = np.eye(size) - np.outer(np.ones(size), u)
        solution, *_ = np.linalg.lstsq(operator, rhs, rcond=None)
        return np.asarray(solution, dtype=float)


class PayoffMatrixDistance(ABC):
    """利得行列間距離の抽象基底クラス。"""

    @abstractmethod
    def calculate(self, left: PayoffMatrix, right: PayoffMatrix) -> float:
        raise NotImplementedError


class MaxElementDifferenceDistance(PayoffMatrixDistance):
    """d(A, B) = max_ij |Aij - Bij|。"""

    def calculate(self, left: PayoffMatrix, right: PayoffMatrix) -> float:
        if left.matrix.shape != right.matrix.shape:
            raise ValueError("matrix shapes must match for distance calculation")
        return float(np.max(np.abs(left.matrix - right.matrix)))


class EquilibriumUStrategyDifferenceDistance(PayoffMatrixDistance):
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


class ApproximationQualityEvaluator:
    """近似器と距離指標を組み合わせて近似精度を評価する。"""

    def __init__(
        self,
        approximation: PayoffMatrixApproximation,
        distance: PayoffMatrixDistance,
    ):
        self.approximation = approximation
        self.distance = distance

    def evaluate(self, source: PayoffMatrix, reference: PayoffMatrix) -> ApproximationResult:
        approximated = self.approximation.approximate(source)
        score = self.distance.calculate(approximated.matrix, reference)
        diagnostics = ApproximationDiagnostics(
            method=approximated.diagnostics.method,
            evaluation=ApproximationEvaluation(quality=score),
        )
        return ApproximationResult(
            matrix=approximated.matrix,
            diagnostics=diagnostics,
        )
