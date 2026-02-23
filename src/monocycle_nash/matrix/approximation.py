from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

import numpy as np

from .base import PayoffMatrix
from .general import GeneralPayoffMatrix
from .monocycle import MonocyclePayoffMatrix

InputMatrixT = TypeVar("InputMatrixT", bound=PayoffMatrix)
ApproxMatrixT = TypeVar("ApproxMatrixT", bound=PayoffMatrix)
ReferenceMatrixT = TypeVar("ReferenceMatrixT", bound=PayoffMatrix)


class PayoffMatrixApproximation(ABC, Generic[InputMatrixT, ApproxMatrixT]):
    """利得行列近似の抽象基底クラス。"""

    @abstractmethod
    def approximate(self, matrix: InputMatrixT) -> ApproxMatrixT:
        raise NotImplementedError


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
