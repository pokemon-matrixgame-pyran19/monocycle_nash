"""実験実行単位と関連ポートのドメイン定義。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class ExperimentRun:
    """実験の1実行を表す最小メタ情報。"""

    serial_id: int
    elapsed_seconds: float
    created_at: datetime
    version: str

    def __post_init__(self) -> None:
        if self.serial_id < 1:
            raise ValueError("serial_id は 1 以上である必要があります")
        if self.elapsed_seconds < 0:
            raise ValueError("elapsed_seconds は 0 以上である必要があります")
        if not self.version.strip():
            raise ValueError("version は空にできません")


class VersionPort(ABC):
    """実行バージョン文字列を取得するポート。"""

    @abstractmethod
    def get_version(self) -> str:
        """現在の実行バージョンを返す。"""
        ...
