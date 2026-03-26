from __future__ import annotations

from monocycle_nash.application_ports import FeatureWorkflowInputPort
import traceback

from monocycle_nash.loader.runtime_common import build_matrix, prepare_run_session, write_input_snapshots
from monocycle_nash.visualization import PayoffDirectedGraphPlotter


FEATURE_NAME = "graph_payoff"


def run(config_loader: FeatureWorkflowInputPort) -> int:
    loaded = config_loader.load_inputs_for_feature(FEATURE_NAME)
    matrix = build_matrix(loaded.matrix_data)
    graph_config = loaded.graph_config
    if graph_config is None:
        raise ValueError("graph 設定が必要です")

    threshold = graph_config.threshold
    canvas_size = graph_config.canvas_size

    service, ctx, conn = prepare_run_session(loaded.setting_data, f"uv run main ({FEATURE_NAME})")
    try:
        write_input_snapshots(
            service,
            ctx.run_id,
            matrix_data=loaded.matrix_data,
            graph_data={"threshold": threshold, "canvas_size": canvas_size},
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
