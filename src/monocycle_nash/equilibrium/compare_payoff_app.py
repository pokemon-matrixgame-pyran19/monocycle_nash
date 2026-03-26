from __future__ import annotations

from monocycle_nash.equilibrium.solve_payoff_app import EquilibriumSettingLoader
from monocycle_nash.loader.main_config import MainConfigLoader


FEATURE_NAME = "compare_payoff"


def run(config_loader: MainConfigLoader) -> int:
    from monocycle_nash.equilibrium.infra import EquilibriumFeatureInfrastructure

    setting_loader: EquilibriumSettingLoader = EquilibriumFeatureInfrastructure(config_loader)
    setting_loader.load_compare_payoff()
    return 0
