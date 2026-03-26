from __future__ import annotations

from monocycle_nash.equilibrium.infra import EquilibriumFeatureInfrastructure
from monocycle_nash.loader.main_config import MainConfigLoader


FEATURE_NAME = "compare_payoff"


def run(config_loader: MainConfigLoader) -> int:
    EquilibriumFeatureInfrastructure(config_loader).load_compare_payoff()
    return 0
