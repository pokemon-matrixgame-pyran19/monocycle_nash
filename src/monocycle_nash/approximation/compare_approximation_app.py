from __future__ import annotations

from monocycle_nash.loader.main_config import MainConfigLoader
import traceback

from monocycle_nash.approximation.infra import ApproximationFeatureInfrastructure
from monocycle_nash.loader.runtime_common import _to_toml, build_matrix, prepare_run_session, write_input_snapshots, write_json
from monocycle_nash.matrix import (
    ApproximationQualityEvaluator,
    DominantEigenpairMonocycleApproximation,
    EquilibriumPreservingResidualMonocycleApproximation,
    EquilibriumUStrategyDifferenceDistance,
    MaxElementDifferenceDistance,
    MonocycleToGeneralApproximation,
    PayoffMatrixApproximation,
    PayoffMatrixDistance,
)


FEATURE_NAME = "compare_approximation"


def run(config_loader: MainConfigLoader) -> int:
    feature_config = ApproximationFeatureInfrastructure(config_loader).load_compare_approximation()
    approximation_config = feature_config.approximation

    service, ctx, conn = prepare_run_session(feature_config.setting_data, f"uv run main ({FEATURE_NAME})")
    try:
        source_matrix_data = feature_config.source_matrix_data
        reference_matrix_data = feature_config.reference_matrix_data

        write_input_snapshots(
            service,
            ctx.run_id,
            matrix_data=source_matrix_data,
            graph_data=None,
            setting_data=feature_config.setting_data,
        )
        input_dir = service.artifact_store.run_dir(ctx.run_id) / "input"
        (input_dir / "reference_matrix.toml").write_text(_to_toml(reference_matrix_data), encoding="utf-8")
        (input_dir / "approximation.toml").write_text(_to_toml(approximation_config.raw_input), encoding="utf-8")

        approximation = _build_approximation(approximation_config.approximation_name)
        distance = _build_distance(approximation_config.distance_name)
        evaluator = ApproximationQualityEvaluator(approximation, distance)

        source_matrix = build_matrix(source_matrix_data)
        reference_matrix = build_matrix(reference_matrix_data)
        score = evaluator.evaluate(source_matrix, reference_matrix)

        write_json(
            service.artifact_store.run_dir(ctx.run_id) / "output" / "approximation_quality.json",
            {
                "source_matrix": approximation_config.source_matrix_name or "<shared.matrix>",
                "reference_matrix": approximation_config.reference_matrix_name or "<shared.matrix>",
                "approximation": approximation.__class__.__name__,
                "distance": distance.__class__.__name__,
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


def _build_approximation(approx_name: str) -> PayoffMatrixApproximation:
    if approx_name == "MonocycleToGeneralApproximation":
        return MonocycleToGeneralApproximation()
    if approx_name == "DominantEigenpairMonocycleApproximation":
        return DominantEigenpairMonocycleApproximation()
    if approx_name == "EquilibriumPreservingResidualMonocycleApproximation":
        return EquilibriumPreservingResidualMonocycleApproximation()
    raise ValueError(f"未対応の approximation です: {approx_name}")


def _build_distance(distance_name: str) -> PayoffMatrixDistance:
    if distance_name == "MaxElementDifferenceDistance":
        return MaxElementDifferenceDistance()
    if distance_name == "EquilibriumUStrategyDifferenceDistance":
        return EquilibriumUStrategyDifferenceDistance()
    raise ValueError(f"未対応の distance です: {distance_name}")


