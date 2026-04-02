from __future__ import annotations

from dataclasses import dataclass
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
    label_values: tuple[ParameterValue, ...]


class ApproximationQualityStatistics:
    """近似精度スコアと入力パラメータを記録し、要約統計量を返す。"""

    def __init__(self, *label_keys: str) -> None:
        if any(not key for key in label_keys):
            raise ValueError("label_keys は空でない文字列で指定してください")
        if len(set(label_keys)) != len(label_keys):
            raise ValueError("label_keys は重複なく指定してください")
        self._label_keys = tuple(label_keys)
        self._records: list[ApproximationQualityRecord] = []

    @property
    def label_keys(self) -> tuple[str, ...]:
        return self._label_keys

    def add(self, score: float, *label_values: ParameterValue) -> None:
        if len(label_values) != len(self._label_keys):
            raise ValueError("label_values の件数は label_keys と一致させてください")
        self._records.append(ApproximationQualityRecord(score=float(score), label_values=tuple(label_values)))

    @property
    def records(self) -> tuple[ApproximationQualityRecord, ...]:
        return tuple(self._records)

    def summarize(self) -> ApproximationQualitySummary:
        return self._summarize_records(self._records)

    def summarize_grouped(self, *group_keys: str) -> dict[tuple[str, ...], ApproximationQualitySummary]:
        if not group_keys:
            raise ValueError("group_keys は1件以上必要です")
        unknown_keys = [key for key in group_keys if key not in self._label_keys]
        if unknown_keys:
            raise ValueError(f"未定義の group_keys です: {unknown_keys}")
        label_indexes = tuple(self._label_keys.index(group_key) for group_key in group_keys)
        bucketed: dict[tuple[str, ...], list[ApproximationQualityRecord]] = {}
        for record in self._records:
            labels = tuple(str(record.label_values[index]) for index in label_indexes)
            bucketed.setdefault(labels, []).append(record)
        return {labels: self._summarize_records(bucket_records) for labels, bucket_records in bucketed.items()}

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
