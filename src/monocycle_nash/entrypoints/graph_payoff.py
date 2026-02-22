from __future__ import annotations

import traceback
from pathlib import Path
from typing import Sequence

from monocycle_nash.visualization import PayoffDirectedGraphPlotter

from .common import (
    build_matrix,
    build_parser,
    load_inputs,
    prepare_run_session,
    write_input_snapshots,
)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser("graph_payoff").parse_args(argv)
    matrix_data, graph_data, setting, _ = load_inputs(
        args.run_config,
        args.data_dir,
        require_graph=True,
        graph_section="payoff",
    )
    matrix = build_matrix(matrix_data)

    if graph_data is None:
        raise ValueError("graph 設定が必要です")
    threshold = float(graph_data.get("threshold", 0.0))
    canvas_size = int(graph_data.get("canvas_size", 840))

    command = f"python -m monocycle_nash.entrypoints.graph_payoff --run-config {args.run_config}"
    service, ctx, conn = prepare_run_session(setting, command)

    try:
        write_input_snapshots(
            service,
            ctx.run_id,
            matrix_data=matrix_data,
            graph_data=graph_data,
            setting_data=setting,
        )
        out_file = service.artifact_store.run_dir(ctx.run_id) / "output" / "edge_graph.svg"
        PayoffDirectedGraphPlotter(matrix.matrix, matrix.labels, threshold=threshold).draw(out_file, canvas_size=canvas_size)
        service.finish_success(ctx, extra_meta={"output_file": "output/edge_graph.svg"})
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
