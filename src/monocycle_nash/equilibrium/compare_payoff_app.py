from __future__ import annotations

from monocycle_nash.application_ports import FeatureWorkflowInputPort


FEATURE_NAME = "compare_payoff"


def run(config_loader: FeatureWorkflowInputPort) -> int:
    config_loader.load_inputs_for_feature(FEATURE_NAME)
    return 0
