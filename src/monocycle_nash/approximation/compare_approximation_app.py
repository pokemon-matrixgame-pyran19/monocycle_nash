from __future__ import annotations

import traceback

from monocycle_nash.loader.data_loader import ExperimentDataLoader
from monocycle_nash.loader.main_config import MainConfigLoader
from monocycle_nash.loader.runtime_common import _to_toml, build_matrix, prepare_run_session, write_input_snapshots, write_json
from monocycle_nash.matrix import ApproximationQualityEvaluator, MaxElementDifferenceDistance, MonocycleToGeneralApproximation


FEATURE_NAME = "compare_approximation"


def run(config_loader: MainConfigLoader) -> int:
    loaded = config_loader.load_inputs_for_feature(FEATURE_NAME)
    approximation_data = loaded.approximation_data
    if approximation_data is None:
        raise ValueError("approximation 設定が必要です")

    service, ctx, conn = prepare_run_session(loaded.setting_data, f"uv run main ({FEATURE_NAME})")
    try:
        source_matrix_data, reference_matrix_data = _load_source_and_reference_matrices(config_loader, loaded.matrix_data, approximation_data)

        write_input_snapshots(
            service,
            ctx.run_id,
            matrix_data=source_matrix_data,
            graph_data=None,
            setting_data=loaded.setting_data,
        )
        input_dir = service.artifact_store.run_dir(ctx.run_id) / "input"
        (input_dir / "reference_matrix.toml").write_text(_to_toml(reference_matrix_data), encoding="utf-8")
        (input_dir / "approximation.toml").write_text(_to_toml(approximation_data), encoding="utf-8")

        approximation = _build_approximation(approximation_data)
        distance = _build_distance(approximation_data)
        evaluator = ApproximationQualityEvaluator(approximation, distance)

        source_matrix = build_matrix(source_matrix_data)
        reference_matrix = build_matrix(reference_matrix_data)
        score = evaluator.evaluate(source_matrix, reference_matrix)

        write_json(
            service.artifact_store.run_dir(ctx.run_id) / "output" / "approximation_quality.json",
            {
                "source_matrix": _resolve_matrix_name(approximation_data, key="source_matrix", default="<shared.matrix>"),
                "reference_matrix": _resolve_matrix_name(approximation_data, key="reference_matrix", default="<shared.matrix>"),
                "approximation": "MonocycleToGeneralApproximation",
                "distance": "MaxElementDifferenceDistance",
                "quality": score,
            },
        )

        service.finish_success(ctx, extra_meta={"output_files": ["output/approximation_quality.json"]})
        return 0
    except Exception as exc:  # noqa: BLE001
        err = traceback.format_exc()
        (service.artifact_store.run_dir(ctx.run_id) / "logs" / "stderr.log").write_text(err, encoding="utf-8")
        service.finish_fail(ctx, extra_meta={"error": str(exc)})
        return 1
    finally:
        conn.close()


def _load_source_and_reference_matrices(
    config_loader: MainConfigLoader,
    default_matrix_data: dict,
    approximation_data: dict,
) -> tuple[dict, dict]:
    base_data_dir = config_loader._main_config_path.parent.parent
    exp_loader = ExperimentDataLoader(base_dir=base_data_dir)

    source_name = _resolve_matrix_name(approximation_data, key="source_matrix")
    reference_name = _resolve_matrix_name(approximation_data, key="reference_matrix")

    source_matrix_data = exp_loader.load("matrix", source_name) if source_name is not None else default_matrix_data
    reference_matrix_data = exp_loader.load("matrix", reference_name) if reference_name is not None else default_matrix_data
    return source_matrix_data, reference_matrix_data


def _build_approximation(approximation_data: dict) -> MonocycleToGeneralApproximation:
    approx_name = _resolve_algorithm_name(approximation_data, kind="approximation")
    if approx_name != "MonocycleToGeneralApproximation":
        raise ValueError(f"未対応の approximation です: {approx_name}")
    return MonocycleToGeneralApproximation()


def _build_distance(approximation_data: dict) -> MaxElementDifferenceDistance:
    distance_name = _resolve_algorithm_name(approximation_data, kind="distance")
    if distance_name != "MaxElementDifferenceDistance":
        raise ValueError(f"未対応の distance です: {distance_name}")
    return MaxElementDifferenceDistance()


def _resolve_algorithm_name(approximation_data: dict, *, kind: str) -> str:
    candidates = [
        approximation_data,
        approximation_data.get("approximation") if isinstance(approximation_data.get("approximation"), dict) else None,
        approximation_data.get("approxmation") if isinstance(approximation_data.get("approxmation"), dict) else None,
    ]
    for container in candidates:
        if container is None:
            continue
        value = container.get(kind)
        if isinstance(value, str) and value:
            return value
    if kind == "approximation":
        return "MonocycleToGeneralApproximation"
    return "MaxElementDifferenceDistance"


def _resolve_matrix_name(approximation_data: dict, *, key: str, default: str | None = None) -> str | None:
    value = approximation_data.get(key)
    if value is None:
        return default
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} は空でない文字列で指定してください")
    return value

