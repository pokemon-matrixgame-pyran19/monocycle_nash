from __future__ import annotations

import traceback

from monocycle_nash.application_support import build_characters, build_matrix, prepare_run_session, write_input_snapshots
from monocycle_nash.visualization import CharacterVectorGraphPlotter, PayoffDirectedGraphPlotter


def run_graph_payoff(*, matrix_data: dict, graph_data: dict, setting_data: dict) -> int:
    matrix = build_matrix(matrix_data)
    threshold = float(graph_data.get("threshold", 0.0))
    canvas_size = int(graph_data.get("canvas_size", 840))

    service, ctx, conn = prepare_run_session(setting_data, "uv run main")
    try:
        write_input_snapshots(service, ctx.run_id, matrix_data=matrix_data, graph_data=graph_data, setting_data=setting_data)
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


def run_plot_characters(*, matrix_data: dict, graph_data: dict, setting_data: dict) -> int:
    characters = build_characters(matrix_data)
    canvas_size = int(graph_data.get("canvas_size", 840))
    margin = int(graph_data.get("margin", 90))

    service, ctx, conn = prepare_run_session(setting_data, "uv run main")
    try:
        write_input_snapshots(service, ctx.run_id, matrix_data=matrix_data, graph_data=graph_data, setting_data=setting_data)
        out_file = service.artifact_store.run_dir(ctx.run_id) / "output" / "character_vector.svg"
        CharacterVectorGraphPlotter(characters).draw(out_file, canvas_size=canvas_size, margin=margin)
        service.finish_success(ctx, extra_meta={"output_file": "output/character_vector.svg"})
        return 0
    except Exception as exc:  # noqa: BLE001
        err = traceback.format_exc()
        (service.artifact_store.run_dir(ctx.run_id) / "logs" / "stderr.log").write_text(err, encoding="utf-8")
        service.finish_fail(ctx, extra_meta={"error": str(exc)})
        return 1
    finally:
        conn.close()
