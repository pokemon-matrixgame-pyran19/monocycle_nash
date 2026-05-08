"""Port interfaces that infrastructure layer can implement.

読み込み対象ごとに個別のポートを定義する。
インフラ層はここで宣言したポートを実装し、アプリケーション層に注入する。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from monocycle_nash.application.node_spec import NodeSpec
from monocycle_nash.application.snapshot import ConfigTreeSnapshot
from monocycle_nash.domain.character import Character
from monocycle_nash.domain.team import Team


class ConfigTreeSnapshotStorePort(ABC):
    """設定ツリースナップショットを永続化するポート。"""

    @abstractmethod
    def store(self, run_id: str, snapshot: ConfigTreeSnapshot) -> Path:
        """スナップショットを保存してファイルパスを返す。"""
        raise NotImplementedError


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
        run_id: str,
        node_path: tuple[str, ...],
        output_method: str,
        filename: str,
    ) -> Path:
        """実行ID・ノード階層・出力方式・ファイル名から保存先パスを返す。"""
        raise NotImplementedError


class MatrixTreeConfigPort(ABC):
    """設定ファイルからノード仕様を読み込むポート。

    インフラ層が実装し、設定 ID またはパスから NodeSpec を返す。
    アプリ層はこのポートを通じてノード設定を取得し、
    MatrixNodeFactory を使って typed ApplicationNode ツリーの行列構築ノードに変換する。
    """

    @abstractmethod
    def load_node_spec(self, config_id: str) -> NodeSpec:
        """設定 ID からノード仕様を読み込んで返す。"""
        raise NotImplementedError
