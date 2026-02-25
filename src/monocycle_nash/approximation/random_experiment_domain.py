from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ApproximationQualitySummary:
    count: int
    minimum: float
    maximum: float
    mean: float
    median: float
    stddev: float


class ApproximationQualityStatistics:
    """近似精度スコア列を記録し、要約統計量を返す。"""

    def __init__(self) -> None:
        self._scores: list[float] = []

    def add(self, score: float) -> None:
        self._scores.append(float(score))

    @property
    def scores(self) -> tuple[float, ...]:
        return tuple(self._scores)

    def summarize(self) -> ApproximationQualitySummary:
        if not self._scores:
            raise ValueError("score が1件以上必要です")

        values = np.asarray(self._scores, dtype=float)
        return ApproximationQualitySummary(
            count=int(values.size),
            minimum=float(np.min(values)),
            maximum=float(np.max(values)),
            mean=float(np.mean(values)),
            median=float(np.median(values)),
            stddev=float(np.std(values)),
        )
