from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from monocycle_nash.loader.toml_tree import TomlTreeLoader


DEFAULT_MAIN_CONFIG_PATH = Path("data/run_config/main.toml")


@dataclass(frozen=True)
class FeatureRunPlan:
    feature: str
    config_path: Path


class MainConfigLoader:
    """main.toml の feature 宣言と feature ごとの設定パス解決のみを扱う。"""

    def __init__(self, main_config_path: Path | str = DEFAULT_MAIN_CONFIG_PATH):
        self._main_config_path = Path(main_config_path)
        self._tree_loader = TomlTreeLoader()

    @property
    def main_config_path(self) -> Path:
        return self._main_config_path

    @property
    def data_root(self) -> Path:
        return self._main_config_path.parent.parent

    def load_features(self) -> list[str]:
        cfg = self._load_main_config()
        features = cfg.get("features")
        if not isinstance(features, list) or not all(isinstance(x, str) and x for x in features):
            raise ValueError("main_config.features は空でない文字列配列で指定してください")
        return features

    def load_feature_run_plans(self) -> list[FeatureRunPlan]:
        return [FeatureRunPlan(feature=feature, config_path=self.resolve_feature_config_path(feature)) for feature in self.load_features()]

    def resolve_feature_config_path(self, feature: str) -> Path:
        cfg = self._load_main_config()
        config_paths = cfg.get("configs")
        if not isinstance(config_paths, dict):
            raise ValueError("main_config.configs はテーブルで指定してください")
        raw_path = config_paths.get(feature)
        if not isinstance(raw_path, str) or not raw_path:
            raise ValueError(f"main_config.configs.{feature} は必須です")
        path = Path(raw_path)
        if not path.is_absolute():
            path = self._main_config_path.parent / path
        return path

    def _load_main_config(self) -> dict[str, Any]:
        return self._tree_loader.load(self._main_config_path)
