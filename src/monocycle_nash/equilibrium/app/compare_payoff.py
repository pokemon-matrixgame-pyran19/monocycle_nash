from __future__ import annotations

from monocycle_nash.equilibrium.app.solve_payoff import EquilibriumSettingLoader
from monocycle_nash.runtime.infra.loader.main_config import MainConfigLoader


FEATURE_NAME = "compare_payoff"


def run(config_loader: MainConfigLoader) -> int:
    from monocycle_nash.equilibrium.infra.feature import EquilibriumFeatureInfrastructure

    setting_loader: EquilibriumSettingLoader = EquilibriumFeatureInfrastructure(config_loader)
    setting_loader.load_compare_payoff()
    return 0
