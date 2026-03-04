from __future__ import annotations

from monocycle_nash.loader.main_config import MainConfigLoader

FEATURE_NAME = "compare_payoff"


def run(config_loader: MainConfigLoader) -> int:
    config_loader.load_inputs_for_feature(FEATURE_NAME)
    return 0
