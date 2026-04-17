"""Port interfaces that infrastructure layer can implement.

読み込み対象ごとに個別のポートを定義する。
インフラ層はここで宣言したポートを実装し、アプリケーション層に注入する。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from monocycle_nash.domain.character import Character
from monocycle_nash.domain.team import Team


class CharacterListFilePort(ABC):
    """ファイルからキャラクターリストを読み込むポート。"""

    @abstractmethod
    def load_characters(self, path: str) -> list[Character]:
        """指定パスのファイルからキャラクターリストを読み込む。"""
        raise NotImplementedError


class TeamListFilePort(ABC):
    """ファイルからチームリストを読み込むポート。"""

    @abstractmethod
    def load_teams(self, path: str) -> list[Team]:
        """指定パスのファイルからチームリストを読み込む。"""
        raise NotImplementedError


class MatrixFilePort(ABC):
    """ファイルから利得行列データを読み込むポート。"""

    @abstractmethod
    def load_matrix(self, path: str) -> np.ndarray:
        """指定パスのファイルから利得行列データを読み込む。"""
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
