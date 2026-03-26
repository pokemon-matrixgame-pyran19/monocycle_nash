from __future__ import annotations

from pathlib import Path

from monocycle_nash.application_ports import (
    ApproximationConfig,
    CharacterGraphConfig,
    LoadedFeatureInputs,
    PayoffGraphConfig,
    RandomMatrixConfig,
)
from monocycle_nash.loader.data_loader import ExperimentDataLoader, SettingDataLoader
from monocycle_nash.loader.runtime_common import validate_setting_input
from monocycle_nash.matrix import MatrixFileInfrastructure
from monocycle_nash.loader.toml_tree import TomlTreeLoader


DEFAULT_MAIN_CONFIG_PATH = Path("data/run_config/main.toml")


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

    def load_inputs_for_feature(self, feature: str) -> LoadedFeatureInputs:
        cfg = self._load_main_config()
        merged = self._resolve_feature_config(cfg, feature)

        matrix_name = self._require_non_empty_str(merged, key="matrix", name=f"{feature}.matrix")
        setting_name = self._require_non_empty_str(merged, key="setting", name=f"{feature}.setting")
        graph_name = self._optional_non_empty_str(merged, key="graph", name=f"{feature}.graph")
        approximation_name = self._resolve_approximation_name(merged, feature=feature)
        random_matrix_name = self._resolve_random_matrix_name(merged, feature=feature)

        data_root = self._main_config_path.parent.parent
        matrix_data = MatrixFileInfrastructure(base_dir=data_root).load_matrix_data(matrix_name)

        exp_loader = ExperimentDataLoader(base_dir=data_root)
        approximation_data = exp_loader.load("approximation", approximation_name) if approximation_name is not None else None
        random_matrix_data = exp_loader.load("random_matrix", random_matrix_name) if random_matrix_name is not None else None
        graph_data = exp_loader.load("graph", graph_name) if graph_name is not None else None

        setting = SettingDataLoader(base_dir=self._main_config_path.parent.parent / "setting").load(setting_name)

        validate_setting_input(setting)

        return LoadedFeatureInputs(
            matrix_data=matrix_data,
            graph_config=self._build_graph_config(feature, graph_data),
            approximation_config=self._build_approximation_config(feature, approximation_data),
            random_matrix_config=self._build_random_matrix_config(feature, random_matrix_data),
            setting_data=setting,
        )

    def load_matrix_data(self, matrix_name: str) -> dict:
        if not isinstance(matrix_name, str) or not matrix_name:
            raise ValueError("matrix_name は空でない文字列で指定してください")
        data_root = self._main_config_path.parent.parent
        return MatrixFileInfrastructure(base_dir=data_root).load_matrix_data(matrix_name)

    def _build_graph_config(self, feature: str, graph_data: dict | None) -> PayoffGraphConfig | CharacterGraphConfig | None:
        if feature == "graph_payoff":
            section = self._require_graph_section(graph_data, "payoff")
            return PayoffGraphConfig(
                threshold=float(section.get("threshold", 0.0)),
                canvas_size=int(section.get("canvas_size", 840)),
            )
        if feature == "plot_characters":
            section = self._require_graph_section(graph_data, "character")
            return CharacterGraphConfig(
                canvas_size=int(section.get("canvas_size", 840)),
                margin=int(section.get("margin", 90)),
            )
        return None

    def _build_approximation_config(self, feature: str, data: dict | None) -> ApproximationConfig | None:
        if feature not in {"compare_approximation", "compare_random_approximation"}:
            return None
        if data is None:
            raise ValueError(f"{feature}.approximation は必須です")

        return ApproximationConfig(
            source_matrix_name=self._optional_non_empty_str(data, key="source_matrix", name="approximation.source_matrix"),
            reference_matrix_name=self._optional_non_empty_str(data, key="reference_matrix", name="approximation.reference_matrix"),
            approximation_name=self._resolve_algorithm_name(data, kind="approximation", default="MonocycleToGeneralApproximation"),
            distance_name=self._resolve_algorithm_name(data, kind="distance", default="MaxElementDifferenceDistance"),
            raw_input=data,
        )

    def _build_random_matrix_config(self, feature: str, data: dict | None) -> RandomMatrixConfig | None:
        if feature != "compare_random_approximation":
            return None
        if data is None:
            raise ValueError("compare_random_approximation.random_matrix は必須です")

        return RandomMatrixConfig(
            size=self._required_int(data, key="size", default=None),
            generation_count=self._required_int(data, key="generation_count", default=100),
            acceptance_condition=self._optional_str(data, key="acceptance_condition", default=""),
            low=self._required_float(data, key="low", default=-1.0),
            high=self._required_float(data, key="high", default=1.0),
            max_attempts=self._required_int(data, key="max_attempts", default=10_000),
            random_seed=self._optional_int(data, key="random_seed"),
            raw_input=data,
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

    @staticmethod
    def _optional_str(container: dict, *, key: str, default: str) -> str:
        value = container.get(key, default)
        if not isinstance(value, str):
            raise ValueError(f"{key} は文字列で指定してください")
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

    @classmethod
    def _resolve_random_matrix_name(cls, container: dict, *, feature: str) -> str | None:
        value = cls._optional_non_empty_str(container, key="random_matrix", name=f"{feature}.random_matrix")
        if value is None and feature == "compare_random_approximation":
            raise ValueError(f"{feature}.random_matrix は必須です")
        return value

    @staticmethod
    def _require_graph_section(graph_data: dict | None, graph_section: str) -> dict:
        if graph_data is None:
            raise ValueError(f"{graph_section} 用の graph 設定が必要です")
        section_data = graph_data.get(graph_section)
        if section_data is None:
            raise ValueError(f"graph.{graph_section} セクションが見つかりません")
        if not isinstance(section_data, dict):
            raise ValueError(f"graph.{graph_section} はテーブルで指定してください")
        return section_data

    @staticmethod
    def _resolve_algorithm_name(data: dict, *, kind: str, default: str) -> str:
        candidates = [
            data,
            data.get("approximation") if isinstance(data.get("approximation"), dict) else None,
            data.get("approxmation") if isinstance(data.get("approxmation"), dict) else None,
        ]
        for container in candidates:
            if container is None:
                continue
            value = container.get(kind)
            if isinstance(value, str) and value:
                return value
        return default

    @staticmethod
    def _required_int(config: dict, *, key: str, default: int | None) -> int:
        raw = config.get(key, default)
        if not isinstance(raw, int):
            raise ValueError(f"random_matrix.{key} は整数で指定してください")
        return raw

    @staticmethod
    def _optional_int(config: dict, *, key: str) -> int | None:
        raw = config.get(key)
        if raw is None:
            return None
        if not isinstance(raw, int):
            raise ValueError(f"random_matrix.{key} は整数で指定してください")
        return raw

    @staticmethod
    def _required_float(config: dict, *, key: str, default: float) -> float:
        raw = config.get(key, default)
        if not isinstance(raw, (int, float)):
            raise ValueError(f"random_matrix.{key} は数値で指定してください")
        return float(raw)
