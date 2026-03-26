from __future__ import annotations

from monocycle_nash.loader.main_config import MainConfigLoader
import traceback

from monocycle_nash.graph.infra import GraphFeatureInfrastructure
from monocycle_nash.loader.runtime_common import (
    build_characters,
    has_matrix_input,
    prepare_run_session,
    write_input_snapshots,
)
from monocycle_nash.visualization import CharacterVectorGraphPlotter


FEATURE_NAME = "plot_characters"


def run(config_loader: MainConfigLoader) -> int:
    feature_config = GraphFeatureInfrastructure(config_loader).load_plot_characters()
    if has_matrix_input(feature_config.matrix_data):
        raise ValueError("plot_characters は matrix ではなく characters 入力が必須です")
    characters = build_characters(feature_config.matrix_data)

    canvas_size = feature_config.canvas_size
    margin = feature_config.margin

    service, ctx, conn = prepare_run_session(feature_config.setting_data, f"uv run main ({FEATURE_NAME})")
    try:
        write_input_snapshots(
            service,
            ctx.run_id,
            matrix_data=feature_config.matrix_data,
            graph_data={"canvas_size": canvas_size, "margin": margin},
            setting_data=feature_config.setting_data,
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
