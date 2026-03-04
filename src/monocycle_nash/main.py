from __future__ import annotations

from monocycle_nash.approximation.compare_approximation_app import run as run_compare_approximation
from monocycle_nash.approximation.compare_random_approximation_app import run as run_compare_random_approximation
from monocycle_nash.equilibrium.compare_payoff_app import run as run_compare_payoff
from monocycle_nash.equilibrium.solve_payoff_app import run as run_solve_payoff
from monocycle_nash.graph.graph_payoff_app import run as run_graph_payoff
from monocycle_nash.graph.plot_characters_app import run as run_plot_characters
from monocycle_nash.loader.main_config import MainConfigLoader


def main() -> int:
    config_loader = MainConfigLoader()
    features = config_loader.load_features()

    runners = {
        "solve_payoff": run_solve_payoff,
        "compare_payoff": run_compare_payoff,
        "compare_approximation": run_compare_approximation,
        "compare_random_approximation": run_compare_random_approximation,
        "graph_payoff": run_graph_payoff,
        "plot_characters": run_plot_characters,
    }

    for feature in features:
        runner = runners.get(feature)
        if runner is None:
            raise ValueError(f"未対応の feature です: {feature}")
        code = runner(config_loader)
        if code != 0:
            return code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
