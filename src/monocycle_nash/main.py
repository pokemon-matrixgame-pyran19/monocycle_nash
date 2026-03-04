from __future__ import annotations

from monocycle_nash.approximation.compare_approximation_app import run as run_compare_approximation
from monocycle_nash.approximation.compare_random_approximation_app import run as run_compare_random_approximation
from monocycle_nash.equilibrium.compare_payoff_app import run as run_compare_payoff
from monocycle_nash.equilibrium.solve_payoff_app import run as run_solve_payoff
from monocycle_nash.graph.graph_payoff_app import run as run_graph_payoff
from monocycle_nash.graph.plot_characters_app import run as run_plot_characters
from monocycle_nash.loader.main_config import MainConfigLoader
from monocycle_nash.loader.runtime_common import finalize_shared_run, set_shared_run_mode


def main() -> int:
    config_loader = MainConfigLoader()
    features = config_loader.load_features()
    shared_mode = len(features) > 1
    set_shared_run_mode(shared_mode)

    runners = {
        "solve_payoff": run_solve_payoff,
        "compare_payoff": run_compare_payoff,
        "compare_approximation": run_compare_approximation,
        "compare_random_approximation": run_compare_random_approximation,
        "graph_payoff": run_graph_payoff,
        "plot_characters": run_plot_characters,
    }

    try:
        for feature in features:
            runner = runners.get(feature)
            if runner is None:
                raise ValueError(f"未対応の feature です: {feature}")
            code = runner(config_loader)
            if code != 0:
                return code
        return 0
    finally:
        if shared_mode:
            finalize_shared_run()
            set_shared_run_mode(False)


if __name__ == "__main__":
    raise SystemExit(main())
