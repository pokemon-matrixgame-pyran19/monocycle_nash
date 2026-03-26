from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
import traceback

from monocycle_nash.loader.main_config import MainConfigLoader
from monocycle_nash.loader.runtime_common import _to_toml, matrix_to_toml_payload, prepare_run_session, write_input_snapshots, write_json
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
from monocycle_nash.matrix.base import PayoffMatrix
from monocycle_nash.runmeta.setting_domain import RuntimeSetting


@dataclass(frozen=True)
class ApproximationSettings:
    source_matrix_name: str | None
    reference_matrix_name: str | None
    approximation_name: str
    distance_name: str
    dominant_eigen_ratio_bin_edges: tuple[float, ...] | None


@dataclass(frozen=True)
class CompareApproximationFeatureConfig:
    matrix: PayoffMatrix
    setting_data: RuntimeSetting
    approximation: ApproximationSettings
    source_matrix: PayoffMatrix
    reference_matrix: PayoffMatrix


class CompareApproximationSettingLoader(ABC):
    @abstractmethod
    def load_compare_approximation(self) -> CompareApproximationFeatureConfig:
        raise NotImplementedError


FEATURE_NAME = "compare_approximation"


def run(config_loader: MainConfigLoader) -> int:
    from monocycle_nash.approximation.infra import ApproximationFeatureInfrastructure

    setting_loader: CompareApproximationSettingLoader = ApproximationFeatureInfrastructure(config_loader)
    feature_config = setting_loader.load_compare_approximation()
    approximation_config = feature_config.approximation

    service, ctx, conn = prepare_run_session(feature_config.setting_data, f"uv run main ({FEATURE_NAME})")
    try:
        source_matrix = feature_config.source_matrix
        reference_matrix = feature_config.reference_matrix

        write_input_snapshots(
            service,
            ctx.run_id,
            matrix_data=matrix_to_toml_payload(source_matrix),
            graph_data=None,
            setting_data=feature_config.setting_data,
        )
        input_dir = service.artifact_store.run_dir(ctx.run_id) / "input"
        (input_dir / "reference_matrix.toml").write_text(
            _to_toml(matrix_to_toml_payload(reference_matrix)),
            encoding="utf-8",
        )
        (input_dir / "approximation.toml").write_text(_to_toml(_approximation_to_toml_payload(approximation_config)), encoding="utf-8")

        approximation = _build_approximation(approximation_config.approximation_name)
        distance = _build_distance(approximation_config.distance_name)
        evaluator = ApproximationQualityEvaluator(approximation, distance)

        result = evaluator.evaluate(source_matrix, reference_matrix)

        write_json(
            service.artifact_store.run_dir(ctx.run_id) / "output" / "approximation_quality.json",
            {
                "source_matrix": approximation_config.source_matrix_name or "<shared.matrix>",
                "reference_matrix": approximation_config.reference_matrix_name or "<shared.matrix>",
                "approximation": approximation.__class__.__name__,
                "distance": distance.__class__.__name__,
                "quality": result.diagnostics.evaluation.quality,
                "diagnostics": {
                    "method": asdict(result.diagnostics.method),
                    "evaluation": asdict(result.diagnostics.evaluation),
                },
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


def _approximation_to_toml_payload(approximation: ApproximationSettings) -> dict[str, object]:
    config = {
        "approximation": approximation.approximation_name,
        "distance": approximation.distance_name,
    }
    if approximation.source_matrix_name is not None:
        config["source_matrix"] = approximation.source_matrix_name
    if approximation.reference_matrix_name is not None:
        config["reference_matrix"] = approximation.reference_matrix_name
    if approximation.dominant_eigen_ratio_bin_edges is not None:
        config["dominant_eigen_ratio_bin_edges"] = list(approximation.dominant_eigen_ratio_bin_edges)
    return config
