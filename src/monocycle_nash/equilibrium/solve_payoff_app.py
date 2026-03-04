from __future__ import annotations

import traceback

import numpy as np

from monocycle_nash.loader.main_config import MainConfigLoader
from monocycle_nash.loader.runtime_common import build_matrix, prepare_run_session, write_input_snapshots, write_json
from monocycle_nash.solver.selector import SolverSelector


FEATURE_NAME = "solve_payoff"


def run(config_loader: MainConfigLoader) -> int:
    loaded = config_loader.load_inputs_for_feature(FEATURE_NAME)
    matrix = build_matrix(loaded.matrix_data)

    service, ctx, conn = prepare_run_session(loaded.setting_data, f"uv run main ({FEATURE_NAME})")
    try:
        write_input_snapshots(
            service,
            ctx.run_id,
            matrix_data=loaded.matrix_data,
            graph_data=None,
            setting_data=loaded.setting_data,
        )

        eq = SolverSelector().solve(matrix)
        out_dir = service.artifact_store.run_dir(ctx.run_id) / "output"

        write_json(
            out_dir / "equilibrium.json",
            {"strategy_ids": eq.strategy_ids, "probabilities": eq.probabilities.tolist()},
        )

        pure_payoff = np.asarray(matrix.matrix, dtype=float) @ np.asarray(eq.probabilities, dtype=float)
        write_json(
            out_dir / "pure_strategy.json",
            {
                "strategy_ids": matrix.row_strategies.ids,
                "labels": matrix.labels,
                "payoffs": pure_payoff.tolist(),
            },
        )

        best = float(np.max(pure_payoff)) if pure_payoff.size else 0.0
        divergence = [float(best - x) for x in pure_payoff.tolist()]
        write_json(
            out_dir / "divergence.json",
            {
                "strategy_ids": matrix.row_strategies.ids,
                "divergence": divergence,
            },
        )

        output_files = [
            "output/equilibrium.json",
            "output/pure_strategy.json",
            "output/divergence.json",
        ]
        if matrix.is_alternating():
            write_json(out_dir / "eigenvalues.json", {"eigenvalues": matrix.eigenvalues().tolist()})
            output_files.append("output/eigenvalues.json")

        service.finish_success(ctx, extra_meta={"output_files": output_files})
        return 0
    except Exception as exc:  # noqa: BLE001
        err = traceback.format_exc()
        (service.artifact_store.run_dir(ctx.run_id) / "logs" / "stderr.log").write_text(err, encoding="utf-8")
        service.finish_fail(ctx, extra_meta={"error": str(exc)})
        return 1
    finally:
        conn.close()
