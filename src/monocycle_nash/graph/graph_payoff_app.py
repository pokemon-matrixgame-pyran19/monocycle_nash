from __future__ import annotations

from pathlib import Path
import traceback

from monocycle_nash.graph.infra import GraphFeatureInfrastructure
from monocycle_nash.loader.runtime_common import matrix_to_toml_payload, prepare_run_session, write_input_snapshots
from monocycle_nash.visualization import PayoffDirectedGraphPlotter


FEATURE_NAME = "graph_payoff"


def run(feature_config_path: Path | str) -> int:
    feature_config = GraphFeatureInfrastructure(feature_config_path).load_graph_payoff()
    matrix = feature_config.matrix

    threshold = feature_config.threshold
    canvas_size = feature_config.canvas_size

    service, ctx, conn = prepare_run_session(feature_config.setting_data, f"uv run main ({FEATURE_NAME})")
    try:
        write_input_snapshots(
            service,
            ctx.run_id,
            matrix_data=matrix_to_toml_payload(matrix),
            graph_data={"threshold": threshold, "canvas_size": canvas_size},
            setting_data=feature_config.setting_data,
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
