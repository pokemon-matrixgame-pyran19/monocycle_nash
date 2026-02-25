from __future__ import annotations

import traceback
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from monocycle_nash.approximation.compare_approximation_app import _build_approximation, _build_distance
from monocycle_nash.approximation.random_experiment_domain import ApproximationQualityStatistics, ApproximationQualitySummary
from monocycle_nash.loader.main_config import MainConfigLoader
from monocycle_nash.loader.runtime_common import (
    _to_toml,
    build_matrix,
    prepare_run_session,
    write_input_snapshots,
    write_json,
)
from monocycle_nash.matrix import ApproximationQualityEvaluator, RandomMatrixAcceptanceCondition, generate_random_skew_symmetric_matrix


FEATURE_NAME = "compare_random_approximation"


@dataclass(frozen=True)
class RandomGenerationConfig:
    size: int
    generation_count: int
    acceptance_condition: RandomMatrixAcceptanceCondition | None
    low: float
    high: float
    max_attempts: int
    random_seed: int | None


class EvenSizeCondition(RandomMatrixAcceptanceCondition):
    def is_satisfied(self, matrix: np.ndarray) -> bool:
        return matrix.shape[0] % 2 == 0


class RankAtLeastFourCondition(RandomMatrixAcceptanceCondition):
    def is_satisfied(self, matrix: np.ndarray) -> bool:
        return int(np.linalg.matrix_rank(matrix)) >= 4


def run(config_loader: MainConfigLoader) -> int:
    loaded = config_loader.load_inputs_for_feature(FEATURE_NAME)
    approximation_data = loaded.approximation_data
    random_matrix_data = loaded.random_matrix_data
    if approximation_data is None:
        raise ValueError("approximation 設定が必要です")
    if random_matrix_data is None:
        raise ValueError("random_matrix 設定が必要です")

    service, ctx, conn = prepare_run_session(loaded.setting_data, f"uv run main ({FEATURE_NAME})")
    try:
        write_input_snapshots(
            service,
            ctx.run_id,
            matrix_data=loaded.matrix_data,
            graph_data=None,
            setting_data=loaded.setting_data,
        )
        input_dir = service.artifact_store.run_dir(ctx.run_id) / "input"
        (input_dir / "approximation.toml").write_text(_to_toml(approximation_data), encoding="utf-8")
        (input_dir / "random_matrix.toml").write_text(_to_toml(random_matrix_data), encoding="utf-8")

        approximation = _build_approximation(approximation_data)
        distance = _build_distance(approximation_data)
        evaluator = ApproximationQualityEvaluator(approximation, distance)

        generation_cfg = _parse_random_generation_config(random_matrix_data)
        rng = np.random.default_rng(generation_cfg.random_seed)

        statistics = ApproximationQualityStatistics()
        for _ in range(generation_cfg.generation_count):
            raw_matrix = generate_random_skew_symmetric_matrix(
                size=generation_cfg.size,
                low=generation_cfg.low,
                high=generation_cfg.high,
                acceptance_condition=generation_cfg.acceptance_condition,
                rng=rng,
                max_attempts=generation_cfg.max_attempts,
            )
            source = build_matrix({"matrix": raw_matrix.tolist()})
            score = evaluator.evaluate(source, source)
            parameters = approximation.quality_parameters(source, config=approximation_data)
            statistics.add(score, parameters=parameters)

        overall = statistics.summarize()
        grouped = _build_grouped_summary(statistics)
        write_json(
            service.artifact_store.run_dir(ctx.run_id) / "output" / "random_approximation_quality.json",
            {
                "generation_count": overall.count,
                "approximation": approximation.__class__.__name__,
                "distance": distance.__class__.__name__,
                "random_matrix": {
                    "size": generation_cfg.size,
                    "low": generation_cfg.low,
                    "high": generation_cfg.high,
                    "max_attempts": generation_cfg.max_attempts,
                    "acceptance_condition": _condition_keyword(generation_cfg.acceptance_condition),
                    "random_seed": generation_cfg.random_seed,
                },
                "quality": asdict(overall),
                "quality_by_parameters": grouped,
            },
        )

        service.finish_success(ctx, extra_meta={"output_files": ["output/random_approximation_quality.json"]})
        return 0
    except Exception as exc:  # noqa: BLE001
        err = traceback.format_exc()
        (service.artifact_store.run_dir(ctx.run_id) / "logs" / "stderr.log").write_text(err, encoding="utf-8")
        service.finish_fail(ctx, extra_meta={"error": str(exc)})
        return 1
    finally:
        conn.close()


def _build_grouped_summary(statistics: ApproximationQualityStatistics) -> dict[str, Any]:
    if not statistics.records:
        return {}
    group_keys = sorted({key for record in statistics.records for key in record.parameters.keys() if key.endswith("_bin")})
    if not group_keys:
        return {}
    grouped = statistics.summarize_grouped(*group_keys)
    return {group_keys[0]: _serialize_grouped_summary(grouped)}


def _serialize_grouped_summary(data: Any) -> Any:
    if isinstance(data, ApproximationQualitySummary):
        return asdict(data)
    if isinstance(data, dict):
        return {key: _serialize_grouped_summary(value) for key, value in data.items()}
    return data


def _condition_keyword(condition: RandomMatrixAcceptanceCondition | None) -> str:
    if condition is None:
        return ""
    if isinstance(condition, EvenSizeCondition):
        return "even_size"
    if isinstance(condition, RankAtLeastFourCondition):
        return "rank_at_least_4"
    raise ValueError("未対応の acceptance_condition です")


def _parse_random_generation_config(config: dict) -> RandomGenerationConfig:
    size = _required_int(config, key="size", default=None)
    generation_count = _required_int(config, key="generation_count", default=100)
    low = _required_float(config, key="low", default=-1.0)
    high = _required_float(config, key="high", default=1.0)
    max_attempts = _required_int(config, key="max_attempts", default=10_000)
    random_seed = _optional_int(config, key="random_seed")

    condition_keyword = config.get("acceptance_condition")
    if condition_keyword is None:
        condition_keyword = ""
    if not isinstance(condition_keyword, str):
        raise ValueError("random_matrix.acceptance_condition は文字列で指定してください")

    acceptance_condition = _build_acceptance_condition(condition_keyword)

    if size <= 0:
        raise ValueError("random_matrix.size は1以上で指定してください")
    if generation_count <= 0:
        raise ValueError("random_matrix.generation_count は1以上で指定してください")

    return RandomGenerationConfig(
        size=size,
        generation_count=generation_count,
        acceptance_condition=acceptance_condition,
        low=low,
        high=high,
        max_attempts=max_attempts,
        random_seed=random_seed,
    )


def _build_acceptance_condition(keyword: str) -> RandomMatrixAcceptanceCondition | None:
    if keyword == "":
        return None
    if keyword == "even_size":
        return EvenSizeCondition()
    if keyword == "rank_at_least_4":
        return RankAtLeastFourCondition()
    raise ValueError(f"未対応の random_matrix.acceptance_condition です: {keyword}")


def _required_int(config: dict, *, key: str, default: int | None) -> int:
    raw = config.get(key, default)
    if not isinstance(raw, int):
        raise ValueError(f"random_matrix.{key} は整数で指定してください")
    return raw


def _optional_int(config: dict, *, key: str) -> int | None:
    raw = config.get(key)
    if raw is None:
        return None
    if not isinstance(raw, int):
        raise ValueError(f"random_matrix.{key} は整数で指定してください")
    return raw


def _required_float(config: dict, *, key: str, default: float) -> float:
    raw = config.get(key, default)
    if not isinstance(raw, (int, float)):
        raise ValueError(f"random_matrix.{key} は数値で指定してください")
    return float(raw)
