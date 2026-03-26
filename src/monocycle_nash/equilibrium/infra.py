from __future__ import annotations

from dataclasses import dataclass

from monocycle_nash.loader.data_loader import SettingDataLoader
from monocycle_nash.loader.main_config import MainConfigLoader
from monocycle_nash.loader.runtime_common import TomlRuntimeSettingParser
from monocycle_nash.matrix import MatrixFileInfrastructure
from monocycle_nash.matrix.base import PayoffMatrix
from monocycle_nash.runmeta.setting_domain import RuntimeSetting


@dataclass(frozen=True)
class SolvePayoffFeatureConfig:
    matrix: PayoffMatrix
    setting_data: RuntimeSetting


class EquilibriumFeatureInfrastructure:
    def __init__(self, config_loader: MainConfigLoader):
        self._config_loader = config_loader
        self._data_root = config_loader.data_root

    def load_compare_payoff(self) -> SolvePayoffFeatureConfig:
        merged = self._config_loader.load_feature_config("compare_payoff")
        matrix_name = _require_non_empty_str(merged, key="matrix", name="compare_payoff.matrix")
        setting_name = _require_non_empty_str(merged, key="setting", name="compare_payoff.setting")

        matrix = MatrixFileInfrastructure(base_dir=self._data_root).load_matrix(matrix_name)
        setting = TomlRuntimeSettingParser().parse(
            SettingDataLoader(base_dir=self._data_root / "setting").load(setting_name)
        )
        return SolvePayoffFeatureConfig(matrix=matrix, setting_data=setting)

    def load_solve_payoff(self) -> SolvePayoffFeatureConfig:
        merged = self._config_loader.load_feature_config("solve_payoff")
        matrix_name = _require_non_empty_str(merged, key="matrix", name="solve_payoff.matrix")
        setting_name = _require_non_empty_str(merged, key="setting", name="solve_payoff.setting")

        matrix = MatrixFileInfrastructure(base_dir=self._data_root).load_matrix(matrix_name)
        setting = TomlRuntimeSettingParser().parse(
            SettingDataLoader(base_dir=self._data_root / "setting").load(setting_name)
        )
        return SolvePayoffFeatureConfig(matrix=matrix, setting_data=setting)


def _require_non_empty_str(container: dict, *, key: str, name: str) -> str:
    value = container.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} は必須です")
    return value
