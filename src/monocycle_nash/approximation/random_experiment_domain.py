from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


ParameterValue = str | int | float | bool


@dataclass(frozen=True)
class ApproximationQualitySummary:
    count: int
    mean: float
    stddev: float


@dataclass(frozen=True)
class ApproximationQualityRecord:
    score: float
    parameters: dict[str, ParameterValue]


class ApproximationQualityStatistics:
    """近似精度スコアと入力パラメータを記録し、要約統計量を返す。"""

    def __init__(self) -> None:
        self._records: list[ApproximationQualityRecord] = []

    def add(self, score: float, *, parameters: dict[str, ParameterValue] | None = None) -> None:
        self._records.append(ApproximationQualityRecord(score=float(score), parameters=dict(parameters or {})))

    @property
    def records(self) -> tuple[ApproximationQualityRecord, ...]:
        return tuple(self._records)

    def summarize(self) -> ApproximationQualitySummary:
        return self._summarize_records(self._records)

    def summarize_grouped(self, *group_keys: str) -> dict[str, Any]:
        if not group_keys:
            raise ValueError("group_keys は1件以上必要です")
        return self._summarize_grouped_recursive(self._records, list(group_keys))

    def _summarize_grouped_recursive(
        self,
        records: list[ApproximationQualityRecord],
        group_keys: list[str],
    ) -> dict[str, Any]:
        current_key = group_keys[0]
        bucketed: dict[str, list[ApproximationQualityRecord]] = {}
        for record in records:
            value = record.parameters.get(current_key, "<missing>")
            bucketed.setdefault(str(value), []).append(record)

        if len(group_keys) == 1:
            return {label: self._summarize_records(bucket_records) for label, bucket_records in bucketed.items()}

        rest_keys = group_keys[1:]
        return {
            label: self._summarize_grouped_recursive(bucket_records, rest_keys)
            for label, bucket_records in bucketed.items()
        }

    @staticmethod
    def _summarize_records(records: list[ApproximationQualityRecord]) -> ApproximationQualitySummary:
        if not records:
            raise ValueError("score が1件以上必要です")
        values = np.asarray([record.score for record in records], dtype=float)
        return ApproximationQualitySummary(
            count=int(values.size),
            mean=float(np.mean(values)),
            stddev=float(np.std(values)),
        )
