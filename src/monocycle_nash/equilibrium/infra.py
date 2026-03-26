from __future__ import annotations

from dataclasses import dataclass

from monocycle_nash.loader.data_loader import SettingDataLoader
from monocycle_nash.loader.main_config import MainConfigLoader
from monocycle_nash.loader.runtime_common import validate_setting_input
from monocycle_nash.matrix import MatrixFileInfrastructure


@dataclass(frozen=True)
class SolvePayoffFeatureConfig:
    matrix_data: dict
    setting_data: dict


class EquilibriumFeatureInfrastructure:
    def __init__(self, config_loader: MainConfigLoader):
        self._config_loader = config_loader
        self._data_root = config_loader.data_root

    def load_compare_payoff(self) -> SolvePayoffFeatureConfig:
        merged = self._config_loader.load_feature_config("compare_payoff")
        matrix_name = _require_non_empty_str(merged, key="matrix", name="compare_payoff.matrix")
        setting_name = _require_non_empty_str(merged, key="setting", name="compare_payoff.setting")

        matrix_data = MatrixFileInfrastructure(base_dir=self._data_root).load_matrix_data(matrix_name)
        setting = SettingDataLoader(base_dir=self._data_root / "setting").load(setting_name)
        validate_setting_input(setting)
        return SolvePayoffFeatureConfig(matrix_data=matrix_data, setting_data=setting)

    def load_solve_payoff(self) -> SolvePayoffFeatureConfig:
        merged = self._config_loader.load_feature_config("solve_payoff")
        matrix_name = _require_non_empty_str(merged, key="matrix", name="solve_payoff.matrix")
        setting_name = _require_non_empty_str(merged, key="setting", name="solve_payoff.setting")

        matrix_data = MatrixFileInfrastructure(base_dir=self._data_root).load_matrix_data(matrix_name)
        setting = SettingDataLoader(base_dir=self._data_root / "setting").load(setting_name)
        validate_setting_input(setting)
        return SolvePayoffFeatureConfig(matrix_data=matrix_data, setting_data=setting)


def _require_non_empty_str(container: dict, *, key: str, name: str) -> str:
    value = container.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} は必須です")
    return value
