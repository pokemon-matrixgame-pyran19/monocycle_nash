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
