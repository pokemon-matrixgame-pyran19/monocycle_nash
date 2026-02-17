from __future__ import annotations

import traceback
from typing import Sequence

import numpy as np

from monocycle_nash.solver.selector import SolverSelector

from .common import build_matrix, build_parser, load_inputs, prepare_run_session, write_input_snapshots, write_json


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser("solve_payoff").parse_args(argv)
    matrix_data, _, setting, _ = load_inputs(args.run_config, args.data_dir, require_graph=False)
    matrix = build_matrix(matrix_data)

    command = f"python -m monocycle_nash.entrypoints.solve_payoff --run-config {args.run_config}"
    service, ctx, conn = prepare_run_session(setting, command)
    try:
        write_input_snapshots(
            service,
            ctx.run_id,
            matrix_data=matrix_data,
            graph_data=None,
            setting_data=setting,
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
            write_json(
                out_dir / "eigenvalues.json",
                {"eigenvalues": matrix.eigenvalues().tolist()},
            )
            output_files.append("output/eigenvalues.json")

        service.finish_success(
            ctx,
            extra_meta={"output_files": output_files},
        )
        return 0
    except Exception as exc:  # noqa: BLE001
        err = traceback.format_exc()
        (service.artifact_store.run_dir(ctx.run_id) / "logs" / "stderr.log").write_text(err, encoding="utf-8")
        service.finish_fail(ctx, extra_meta={"error": str(exc)})
        return 1
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
