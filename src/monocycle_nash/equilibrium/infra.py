from __future__ import annotations

from pathlib import Path

from monocycle_nash.equilibrium.solve_payoff_app import EquilibriumSettingLoader, SolvePayoffFeatureConfig
from monocycle_nash.loader.data_loader import SettingDataLoader
from monocycle_nash.loader.runtime_common import TomlRuntimeSettingParser
from monocycle_nash.loader.toml_tree import TomlTreeLoader
from monocycle_nash.matrix import MatrixFileInfrastructure


class EquilibriumFeatureInfrastructure(EquilibriumSettingLoader):
    def __init__(self, feature_config_path: Path | str):
        self._config_path = Path(feature_config_path)
        self._data_root = self._config_path.parent.parent
        self._tree_loader = TomlTreeLoader()

    def load_compare_payoff(self) -> SolvePayoffFeatureConfig:
        merged = self._load_feature_config()
        matrix_name = _require_non_empty_str(merged, key="matrix", name="compare_payoff.matrix")
        setting_name = _require_non_empty_str(merged, key="setting", name="compare_payoff.setting")

        matrix = MatrixFileInfrastructure(base_dir=self._data_root).load_matrix(matrix_name)
        setting = TomlRuntimeSettingParser().parse(
            SettingDataLoader(base_dir=self._data_root / "setting").load(setting_name)
        )
        return SolvePayoffFeatureConfig(matrix=matrix, setting_data=setting)

    def load_solve_payoff(self) -> SolvePayoffFeatureConfig:
        merged = self._load_feature_config()
        matrix_name = _require_non_empty_str(merged, key="matrix", name="solve_payoff.matrix")
        setting_name = _require_non_empty_str(merged, key="setting", name="solve_payoff.setting")

        matrix = MatrixFileInfrastructure(base_dir=self._data_root).load_matrix(matrix_name)
        setting = TomlRuntimeSettingParser().parse(
            SettingDataLoader(base_dir=self._data_root / "setting").load(setting_name)
        )
        return SolvePayoffFeatureConfig(matrix=matrix, setting_data=setting)

    def _load_feature_config(self) -> dict:
        return self._tree_loader.load(self._config_path)


def _require_non_empty_str(container: dict, *, key: str, name: str) -> str:
    value = container.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} は必須です")
    return value
