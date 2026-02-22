from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from monocycle_nash.loader.data_loader import ExperimentDataLoader, SettingDataLoader
from monocycle_nash.loader.toml_tree import TomlTreeLoader

MAIN_RUN_CONFIG_PATH = Path("data/run_config/main.toml")


@dataclass(frozen=True)
class FeatureInputRefs:
    matrix: str
    setting: str
    graph: str | None


class RunConfigLoader:
    def __init__(self, data_dir: Path | str = "data", tree_loader: TomlTreeLoader | None = None):
        self._data_dir = Path(data_dir)
        self._tree_loader = tree_loader or TomlTreeLoader()

    def load_main(self, config_path: Path | str = MAIN_RUN_CONFIG_PATH) -> dict[str, Any]:
        return self._tree_loader.load(Path(config_path))

    def enabled_features(self, *, config_path: Path | str = MAIN_RUN_CONFIG_PATH) -> list[str]:
        cfg = self.load_main(config_path)
        features = cfg.get("features")
        if not isinstance(features, list) or not features or not all(isinstance(v, str) and v for v in features):
            raise ValueError("main.features は空でない文字列配列で指定してください")
        return features

    def load_feature_inputs(
        self,
        feature_name: str,
        *,
        requires_graph: bool,
        config_path: Path | str = MAIN_RUN_CONFIG_PATH,
    ) -> tuple[dict[str, Any], dict[str, Any] | None, dict[str, Any]]:
        refs = self.load_feature_refs(feature_name, config_path=config_path)
        if requires_graph and refs.graph is None:
            raise ValueError(f"{feature_name} には graph 設定が必要です")

        exp_loader = ExperimentDataLoader(base_dir=self._data_dir)
        matrix_data = exp_loader.load("matrix", refs.matrix)
        graph_data = exp_loader.load("graph", refs.graph) if refs.graph is not None else None
        setting_data = SettingDataLoader(base_dir=self._data_dir / "setting").load(refs.setting)
        return matrix_data, graph_data, setting_data

    def load_feature_refs(self, feature_name: str, *, config_path: Path | str = MAIN_RUN_CONFIG_PATH) -> FeatureInputRefs:
        cfg = self.load_main(config_path)
        shared = cfg.get("shared", {})
        feature_section = cfg.get(feature_name, {})

        if not isinstance(shared, dict):
            raise ValueError("main.shared はテーブルで指定してください")
        if not isinstance(feature_section, dict):
            raise ValueError(f"main.{feature_name} はテーブルで指定してください")

        merged = {**shared, **feature_section}
        matrix = self._require_non_empty_str(merged, "matrix")
        setting = self._require_non_empty_str(merged, "setting")
        graph = self._optional_graph_ref(merged)
        return FeatureInputRefs(matrix=matrix, setting=setting, graph=graph)

    @staticmethod
    def _require_non_empty_str(cfg: dict[str, Any], key: str) -> str:
        value = cfg.get(key)
        if value is None:
            raise ValueError(f"run_config.{key} は必須の文字列です")
        if not isinstance(value, str) or not value:
            raise ValueError(f"run_config.{key} は空でない文字列で指定してください")
        return value

    @staticmethod
    def _optional_graph_ref(cfg: dict[str, Any]) -> str | None:
        graph = cfg.get("graph")
        if graph is None:
            return None
        if not isinstance(graph, str):
            raise ValueError("run_config.graph は文字列で指定してください")
        if graph == "":
            return None
        return graph
