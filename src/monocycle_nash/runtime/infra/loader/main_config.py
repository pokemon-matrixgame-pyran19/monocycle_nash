from __future__ import annotations

from pathlib import Path
from typing import Any

from monocycle_nash.runtime.infra.loader.toml_tree import TomlTreeLoader


DEFAULT_MAIN_CONFIG_PATH = Path("data/run_config/main.toml")


class MainConfigLoader:
    """main.toml の feature 宣言と shared/feature マージのみを扱う。"""

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

    def load_feature_config(self, feature: str) -> dict[str, Any]:
        cfg = self._load_main_config()
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

        merged: dict[str, Any] = {}
        merged.update(shared)
        merged.update(feature_table)
        return merged

    def _load_main_config(self) -> dict[str, Any]:
        return self._tree_loader.load(self._main_config_path)
