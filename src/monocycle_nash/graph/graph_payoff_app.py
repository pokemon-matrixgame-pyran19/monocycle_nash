from __future__ import annotations

import traceback

from monocycle_nash.loader.main_config import MainConfigLoader
from monocycle_nash.loader.runtime_common import build_matrix, prepare_run_session, write_input_snapshots
from monocycle_nash.visualization import PayoffDirectedGraphPlotter


FEATURE_NAME = "graph_payoff"


def run(config_loader: MainConfigLoader) -> int:
    loaded = config_loader.load_inputs_for_feature(FEATURE_NAME, graph_section="payoff")
    matrix = build_matrix(loaded.matrix_data)
    graph_data = loaded.graph_data
    if graph_data is None:
        raise ValueError("graph 設定が必要です")

    threshold = float(graph_data.get("threshold", 0.0))
    canvas_size = int(graph_data.get("canvas_size", 840))

    service, ctx, conn = prepare_run_session(loaded.setting_data, f"uv run main ({FEATURE_NAME})")
    try:
        write_input_snapshots(
            service,
            ctx.run_id,
            matrix_data=loaded.matrix_data,
            graph_data=graph_data,
            setting_data=loaded.setting_data,
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
