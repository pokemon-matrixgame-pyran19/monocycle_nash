"""Port interfaces (ABCs) that infrastructure will implement."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class RunContext:
    """実行コンテキスト - 結果の出力先を管理"""

    run_id: str
    base_dir: Path

    @property
    def input_dir(self) -> Path:
        return self.base_dir / self.run_id / "input"

    @property
    def artifact_dir(self) -> Path:
        return self.base_dir / self.run_id / "artifact"

    @property
    def metadata_path(self) -> Path:
        return self.base_dir / self.run_id / "metadata.json"


class MatrixDataPort(ABC):
    """行列データの読み込みポート"""

    @abstractmethod
    def load_matrix_data(self, identifier: str) -> dict:
        """data/matrix/<identifier>/data.toml を読み込む"""
        ...


class ExperimentDataPort(ABC):
    """実験データの読み込みポート"""

    @abstractmethod
    def load_experiment_data(self, class_name: str, identifier: str) -> dict:
        """data/<class_name>/<identifier>/data.toml を読み込む"""
        ...


class GraphConfigPort(ABC):
    """グラフ設定の読み込みポート"""

    @abstractmethod
    def load_graph_config(self, identifier: str) -> dict:
        """data/graph/<identifier>/data.toml を読み込む"""
        ...


class OutputPort(ABC):
    """結果の出力ポート"""

    @abstractmethod
    def create_run_context(self, use_case_name: str) -> RunContext:
        """新しい実行コンテキストを生成し、ディレクトリを作成"""
        ...

    @abstractmethod
    def write_json(self, ctx: RunContext, filename: str, data: dict) -> None:
        """artifact_dir にJSONファイルを書き出す"""
        ...

    @abstractmethod
    def write_svg(self, ctx: RunContext, filename: str, content: str) -> None:
        """artifact_dir にSVGファイルを書き出す"""
        ...

    @abstractmethod
    def save_input_snapshot(self, ctx: RunContext, filename: str, data: dict) -> None:
        """input_dir に入力データのスナップショットを保存"""
        ...

    @abstractmethod
    def write_metadata(self, ctx: RunContext, metadata: dict) -> None:
        """metadata.json に実行メタデータを書き出す"""
        ...


class VisualizationPort(ABC):
    """可視化ポート"""

    @abstractmethod
    def draw_payoff_graph(
        self,
        matrix: np.ndarray,
        labels: list[str],
        threshold: float,
        canvas_size: int,
    ) -> str:
        """利得行列のグラフをSVG文字列で返す"""
        ...

    @abstractmethod
    def draw_character_plot(
        self,
        characters: list,
        canvas_size: int,
        margin: int,
    ) -> str:
        """キャラクターベクトルプロットをSVG文字列で返す"""
        ...
