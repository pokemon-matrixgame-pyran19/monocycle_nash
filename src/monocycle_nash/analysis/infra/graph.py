from __future__ import annotations

from dataclasses import dataclass

from monocycle_nash.runtime.infra.loader.data_loader import ExperimentDataLoader, SettingDataLoader
from monocycle_nash.runtime.infra.loader.main_config import MainConfigLoader
from monocycle_nash.game.infra.matrix import build_characters
from monocycle_nash.runtime.infra.loader.runtime_common import TomlRuntimeSettingParser
from monocycle_nash.game.infra.matrix import MatrixFileInfrastructure
from monocycle_nash.game.domain.matrix.base import PayoffMatrix
from monocycle_nash.game.domain.character import Character
from monocycle_nash.runtime.infra.runmeta.setting_domain import RuntimeSetting


@dataclass(frozen=True)
class GraphPayoffFeatureConfig:
    matrix: PayoffMatrix
    setting_data: RuntimeSetting
    threshold: float
    canvas_size: int


@dataclass(frozen=True)
class PlotCharactersFeatureConfig:
    characters: list[Character]
    setting_data: RuntimeSetting
    canvas_size: int
    margin: int


class GraphFeatureInfrastructure:
    def __init__(self, config_loader: MainConfigLoader):
        self._config_loader = config_loader
        self._data_root = config_loader.data_root

    def load_graph_payoff(self) -> GraphPayoffFeatureConfig:
        merged = self._config_loader.load_feature_config("graph_payoff")
        matrix_name = _require_non_empty_str(merged, key="matrix", name="graph_payoff.matrix")
        setting_name = _require_non_empty_str(merged, key="setting", name="graph_payoff.setting")
        graph_name = _require_non_empty_str(merged, key="graph", name="graph_payoff.graph")

        matrix = MatrixFileInfrastructure(base_dir=self._data_root).load_matrix(matrix_name)
        setting = TomlRuntimeSettingParser().parse(
            SettingDataLoader(base_dir=self._data_root / "setting").load(setting_name)
        )

        graph_data = ExperimentDataLoader(base_dir=self._data_root).load("graph", graph_name)
        section = _require_graph_section(graph_data, "payoff")
        return GraphPayoffFeatureConfig(
            matrix=matrix,
            setting_data=setting,
            threshold=float(section.get("threshold", 0.0)),
            canvas_size=int(section.get("canvas_size", 840)),
        )

    def load_plot_characters(self) -> PlotCharactersFeatureConfig:
        merged = self._config_loader.load_feature_config("plot_characters")
        matrix_name = _require_non_empty_str(merged, key="matrix", name="plot_characters.matrix")
        setting_name = _require_non_empty_str(merged, key="setting", name="plot_characters.setting")
        graph_name = _require_non_empty_str(merged, key="graph", name="plot_characters.graph")

        matrix_input = MatrixFileInfrastructure(base_dir=self._data_root).load_matrix_input(matrix_name)
        characters = build_characters(matrix_input)
        setting = TomlRuntimeSettingParser().parse(
            SettingDataLoader(base_dir=self._data_root / "setting").load(setting_name)
        )

        graph_data = ExperimentDataLoader(base_dir=self._data_root).load("graph", graph_name)
        section = _require_graph_section(graph_data, "character")
        return PlotCharactersFeatureConfig(
            characters=characters,
            setting_data=setting,
            canvas_size=int(section.get("canvas_size", 840)),
            margin=int(section.get("margin", 90)),
        )


def _require_non_empty_str(container: dict, *, key: str, name: str) -> str:
    value = container.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} は必須です")
    return value


def _require_graph_section(graph_data: dict | None, graph_section: str) -> dict:
    if graph_data is None:
        raise ValueError(f"{graph_section} 用の graph 設定が必要です")
    section_data = graph_data.get(graph_section)
    if section_data is None:
        raise ValueError(f"graph.{graph_section} セクションが見つかりません")
    if not isinstance(section_data, dict):
        raise ValueError(f"graph.{graph_section} はテーブルで指定してください")
    return section_data
