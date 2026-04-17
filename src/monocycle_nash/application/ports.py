"""Port interfaces that infrastructure layer can implement."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Mapping


class ConfigReferencePort(ABC):
    """設定参照（例: ファイル参照）を解決するポート。"""

    @abstractmethod
    def load_config(self, reference: str) -> Mapping[str, Any]:
        """参照文字列に対応する設定データを返す。"""
        raise NotImplementedError


class OutputPathPort(ABC):
    """出力ファイルの保存先パスを解決するポート。"""

    @abstractmethod
    def resolve_output_path(
        self,
        *,
        node_name: str,
        output_method: str,
        filename: str,
    ) -> Path:
        """ノード名・出力方式・ファイル名から保存先パスを返す。"""
        raise NotImplementedError
