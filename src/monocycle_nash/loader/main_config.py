from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from monocycle_nash.loader.data_loader import ExperimentDataLoader, SettingDataLoader
from monocycle_nash.loader.runtime_common import validate_graph_input, validate_matrix_input, validate_setting_input
from monocycle_nash.loader.toml_tree import TomlTreeLoader


DEFAULT_MAIN_CONFIG_PATH = Path("data/run_config/main.toml")


@dataclass(frozen=True)
class LoadedFeatureInputs:
    matrix_data: dict
    graph_data: dict | None
    approximation_data: dict | None
    setting_data: dict


class MainConfigLoader:
    def __init__(self, main_config_path: Path | str = DEFAULT_MAIN_CONFIG_PATH):
        self._main_config_path = Path(main_config_path)
        self._tree_loader = TomlTreeLoader()

    def load_features(self) -> list[str]:
        cfg = self._load_main_config()
        features = cfg.get("features")
        if not isinstance(features, list) or not all(isinstance(x, str) and x for x in features):
            raise ValueError("main_config.features は空でない文字列配列で指定してください")
        return features

    def load_inputs_for_feature(self, feature: str, *, graph_section: str | None = None) -> LoadedFeatureInputs:
        cfg = self._load_main_config()
        merged = self._resolve_feature_config(cfg, feature)

        matrix_name = self._require_non_empty_str(merged, key="matrix", name=f"{feature}.matrix")
        setting_name = self._require_non_empty_str(merged, key="setting", name=f"{feature}.setting")
        graph_name = self._optional_non_empty_str(merged, key="graph", name=f"{feature}.graph")
        approximation_name = self._resolve_approximation_name(merged, feature=feature)

        exp_loader = ExperimentDataLoader(base_dir=self._main_config_path.parent.parent)
        matrix_data = exp_loader.load("matrix", matrix_name)
        loaded_graph = exp_loader.load("graph", graph_name) if graph_name is not None else None
        approximation_data = exp_loader.load("approximation", approximation_name) if approximation_name is not None else None
        graph_data = loaded_graph if graph_section is None else self._select_graph_section(loaded_graph, graph_section=graph_section)

        setting = SettingDataLoader(base_dir=self._main_config_path.parent.parent / "setting").load(setting_name)

        validate_matrix_input(matrix_data)
        validate_graph_input(graph_data)
        validate_setting_input(setting)
        return LoadedFeatureInputs(
            matrix_data=matrix_data,
            graph_data=graph_data,
            approximation_data=approximation_data,
            setting_data=setting,
        )

    def _load_main_config(self) -> dict:
        return self._tree_loader.load(self._main_config_path)

    def _resolve_feature_config(self, cfg: dict, feature: str) -> dict:
        shared = cfg.get("shared")
        if shared is None:
            shared = {}
        if not isinstance(shared, dict):
            raise ValueError("main_config.shared はテーブルで指定してください")

        feature_table = cfg.get(feature)
        if feature_table is None:
            raise ValueError(f"main_config.{feature} セクションが見つかりません")
        if not isinstance(feature_table, dict):
            raise ValueError(f"main_config.{feature} はテーブルで指定してください")

        merged: dict = {}
        merged.update(shared)
        merged.update(feature_table)
        return merged

    @staticmethod
    def _optional_non_empty_str(container: dict, *, key: str, name: str) -> str | None:
        value = container.get(key)
        if value is None:
            return None
        if not isinstance(value, str) or not value:
            raise ValueError(f"{name} は空でない文字列で指定してください")
        return value

    @classmethod
    def _require_non_empty_str(cls, container: dict, *, key: str, name: str) -> str:
        value = cls._optional_non_empty_str(container, key=key, name=name)
        if value is None:
            raise ValueError(f"{name} は必須です")
        return value

    @classmethod
    def _resolve_approximation_name(cls, container: dict, *, feature: str) -> str | None:
        value = cls._optional_non_empty_str(container, key="approximation", name=f"{feature}.approximation")
        if value is None and feature.startswith("compare_"):
            raise ValueError(f"{feature}.approximation は必須です")
        return value

    @staticmethod
    def _select_graph_section(graph_data: dict | None, *, graph_section: str) -> dict:
        if graph_data is None:
            raise ValueError(f"{graph_section} 用の graph 設定が必要です")
        section_data = graph_data.get(graph_section)
        if section_data is None:
            raise ValueError(f"graph.{graph_section} セクションが見つかりません")
        if not isinstance(section_data, dict):
            raise ValueError(f"graph.{graph_section} はテーブルで指定してください")
        return section_data
