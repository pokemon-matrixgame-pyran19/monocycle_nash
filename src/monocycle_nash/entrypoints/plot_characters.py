from __future__ import annotations

import traceback
from typing import Sequence

from monocycle_nash.character import Character
from monocycle_nash.visualization import CharacterVectorGraphPlotter

from .common import build_parser, load_inputs, prepare_run_session, write_input_snapshots


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser("plot_characters").parse_args(argv)
    matrix_data, graph_data, setting, _ = load_inputs(args.run_config, args.data_dir, require_graph=True)

    chars_raw = matrix_data.get("characters")
    if not isinstance(chars_raw, list) or not chars_raw:
        raise ValueError("plot_characters は matrix ではなく characters 入力が必須です")
    characters: list[Character] = []
    for idx, item in enumerate(chars_raw):
        if not isinstance(item, dict):
            raise ValueError(f"characters[{idx}] はテーブルで指定してください")
        label = item.get("label")
        power = item.get("p")
        vector = item.get("v")
        if not isinstance(label, str) or not isinstance(power, (int, float)) or not isinstance(vector, list) or len(vector) != 2:
            raise ValueError(f"characters[{idx}] の形式が不正です")
        characters.append(Character(float(power), vector, label=label))

    canvas_size = int((graph_data or {}).get("canvas_size", 840))
    margin = int((graph_data or {}).get("margin", 90))

    command = f"python -m monocycle_nash.entrypoints.plot_characters --run-config {args.run_config}"
    service, ctx, conn = prepare_run_session(setting, command)
    try:
        write_input_snapshots(
            service,
            ctx.run_id,
            matrix_data=matrix_data,
            graph_data=graph_data,
            setting_data=setting,
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


if __name__ == "__main__":
    raise SystemExit(main())
