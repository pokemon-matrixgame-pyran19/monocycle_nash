from __future__ import annotations

from dataclasses import dataclass

from monocycle_nash.loader.data_loader import ExperimentDataLoader, SettingDataLoader
from monocycle_nash.loader.main_config import MainConfigLoader
from monocycle_nash.loader.runtime_common import validate_setting_input
from monocycle_nash.matrix import MatrixFileInfrastructure


@dataclass(frozen=True)
class GraphPayoffFeatureConfig:
    matrix_data: dict
    setting_data: dict
    threshold: float
    canvas_size: int


@dataclass(frozen=True)
class PlotCharactersFeatureConfig:
    matrix_data: dict
    setting_data: dict
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

        matrix_data = MatrixFileInfrastructure(base_dir=self._data_root).load_matrix_data(matrix_name)
        setting = SettingDataLoader(base_dir=self._data_root / "setting").load(setting_name)
        validate_setting_input(setting)

        graph_data = ExperimentDataLoader(base_dir=self._data_root).load("graph", graph_name)
        section = _require_graph_section(graph_data, "payoff")
        return GraphPayoffFeatureConfig(
            matrix_data=matrix_data,
            setting_data=setting,
            threshold=float(section.get("threshold", 0.0)),
            canvas_size=int(section.get("canvas_size", 840)),
        )

    def load_plot_characters(self) -> PlotCharactersFeatureConfig:
        merged = self._config_loader.load_feature_config("plot_characters")
        matrix_name = _require_non_empty_str(merged, key="matrix", name="plot_characters.matrix")
        setting_name = _require_non_empty_str(merged, key="setting", name="plot_characters.setting")
        graph_name = _require_non_empty_str(merged, key="graph", name="plot_characters.graph")

        matrix_data = MatrixFileInfrastructure(base_dir=self._data_root).load_matrix_data(matrix_name)
        setting = SettingDataLoader(base_dir=self._data_root / "setting").load(setting_name)
        validate_setting_input(setting)

        graph_data = ExperimentDataLoader(base_dir=self._data_root).load("graph", graph_name)
        section = _require_graph_section(graph_data, "character")
        return PlotCharactersFeatureConfig(
            matrix_data=matrix_data,
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
