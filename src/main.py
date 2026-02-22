from __future__ import annotations

from monocycle_nash.equilibrium.application import run_solve_payoff
from monocycle_nash.graph.application import run_graph_payoff, run_plot_characters
from monocycle_nash.loader import RunConfigLoader


def main() -> int:
    loader = RunConfigLoader()
    exit_code = 0
    for feature in loader.enabled_features():
        if feature == "solve_payoff":
            matrix_data, _, setting_data = loader.load_feature_inputs(feature, requires_graph=False)
            code = run_solve_payoff(matrix_data=matrix_data, setting_data=setting_data)
        elif feature == "graph_payoff":
            matrix_data, graph_data, setting_data = loader.load_feature_inputs(feature, requires_graph=True)
            assert graph_data is not None
            code = run_graph_payoff(matrix_data=matrix_data, graph_data=graph_data, setting_data=setting_data)
        elif feature == "plot_characters":
            matrix_data, graph_data, setting_data = loader.load_feature_inputs(feature, requires_graph=True)
            assert graph_data is not None
            code = run_plot_characters(matrix_data=matrix_data, graph_data=graph_data, setting_data=setting_data)
        else:
            raise ValueError(f"未対応のfeatureです: {feature}")

        if code != 0:
            exit_code = code
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
