from __future__ import annotations

from pathlib import Path

from monocycle_nash.equilibrium.solve_payoff_app import EquilibriumSettingLoader


FEATURE_NAME = "compare_payoff"


def run(feature_config_path: Path | str) -> int:
    from monocycle_nash.equilibrium.infra import EquilibriumFeatureInfrastructure

    setting_loader: EquilibriumSettingLoader = EquilibriumFeatureInfrastructure(feature_config_path)
    setting_loader.load_compare_payoff()
    return 0
