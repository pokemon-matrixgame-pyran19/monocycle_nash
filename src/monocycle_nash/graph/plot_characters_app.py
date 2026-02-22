from __future__ import annotations

import traceback

from monocycle_nash.loader.main_config import MainConfigLoader
from monocycle_nash.loader.runtime_common import (
    build_characters,
    has_matrix_input,
    prepare_run_session,
    write_input_snapshots,
)
from monocycle_nash.visualization import CharacterVectorGraphPlotter


FEATURE_NAME = "plot_characters"


def run(config_loader: MainConfigLoader) -> int:
    loaded = config_loader.load_inputs_for_feature(FEATURE_NAME, graph_section="character")
    if has_matrix_input(loaded.matrix_data):
        raise ValueError("plot_characters は matrix ではなく characters 入力が必須です")
    characters = build_characters(loaded.matrix_data)
    graph_data = loaded.graph_data or {}

    canvas_size = int(graph_data.get("canvas_size", 840))
    margin = int(graph_data.get("margin", 90))

    service, ctx, conn = prepare_run_session(loaded.setting_data, f"uv run main ({FEATURE_NAME})")
    try:
        write_input_snapshots(
            service,
            ctx.run_id,
            matrix_data=loaded.matrix_data,
            graph_data=loaded.graph_data,
            setting_data=loaded.setting_data,
        )
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
