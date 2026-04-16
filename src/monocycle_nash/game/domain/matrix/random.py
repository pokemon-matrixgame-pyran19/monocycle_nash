from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class RandomMatrixAcceptanceCondition(ABC):
    """ランダム行列の採用可否を判定する抽象クラス。"""

    @abstractmethod
    def is_satisfied(self, matrix: np.ndarray) -> bool:
        """条件を満たす場合はTrueを返す。"""


def generate_random_skew_symmetric_matrix(
    size: int,
    low: float = -1.0,
    high: float = 1.0,
    acceptance_condition: RandomMatrixAcceptanceCondition | None = None,
    *,
    rng: np.random.Generator | None = None,
    max_attempts: int = 10_000,
) -> np.ndarray:
    """
    対角成分0・非対角成分が交代行列のランダム行列を生成する。

    acceptance_condition が指定された場合は満たすまで再生成する。
    """
    if size <= 0:
        raise ValueError("size は1以上である必要があります")
    if high <= low:
        raise ValueError("high は low より大きい必要があります")
    if max_attempts <= 0:
        raise ValueError("max_attempts は1以上である必要があります")

    random = rng if rng is not None else np.random.default_rng()

    for _ in range(max_attempts):
        matrix = np.zeros((size, size), dtype=float)
        upper_indices = np.triu_indices(size, k=1)
        values = random.uniform(low, high, size=upper_indices[0].size)
        matrix[upper_indices] = values
        matrix[(upper_indices[1], upper_indices[0])] = -values

        if acceptance_condition is None or acceptance_condition.is_satisfied(matrix):
            return matrix

    raise RuntimeError("指定された条件を満たす行列を生成できませんでした")

