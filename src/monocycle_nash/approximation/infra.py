from __future__ import annotations

from dataclasses import dataclass

from monocycle_nash.loader.data_loader import ExperimentDataLoader, SettingDataLoader
from monocycle_nash.loader.main_config import MainConfigLoader
from monocycle_nash.loader.runtime_common import validate_setting_input
from monocycle_nash.matrix import MatrixFileInfrastructure
from monocycle_nash.matrix.base import PayoffMatrix


@dataclass(frozen=True)
class ApproximationSettings:
    source_matrix_name: str | None
    reference_matrix_name: str | None
    approximation_name: str
    distance_name: str
    raw_input: dict


@dataclass(frozen=True)
class RandomMatrixSettings:
    size: int
    generation_count: int
    acceptance_condition: str
    low: float
    high: float
    max_attempts: int
    random_seed: int | None
    raw_input: dict


@dataclass(frozen=True)
class CompareApproximationFeatureConfig:
    matrix: PayoffMatrix
    setting_data: dict
    approximation: ApproximationSettings
    source_matrix: PayoffMatrix
    reference_matrix: PayoffMatrix


@dataclass(frozen=True)
class CompareRandomApproximationFeatureConfig:
    matrix: PayoffMatrix
    setting_data: dict
    approximation: ApproximationSettings
    random_matrix: RandomMatrixSettings


class ApproximationFeatureInfrastructure:
    def __init__(self, config_loader: MainConfigLoader):
        self._config_loader = config_loader
        self._data_root = config_loader.data_root

    def load_compare_approximation(self) -> CompareApproximationFeatureConfig:
        merged = self._config_loader.load_feature_config("compare_approximation")
        matrix_name = _require_non_empty_str(merged, key="matrix", name="compare_approximation.matrix")
        setting_name = _require_non_empty_str(merged, key="setting", name="compare_approximation.setting")
        approximation_name = _require_non_empty_str(merged, key="approximation", name="compare_approximation.approximation")

        matrix_repo = MatrixFileInfrastructure(base_dir=self._data_root)
        matrix = matrix_repo.load_matrix(matrix_name)
        setting = SettingDataLoader(base_dir=self._data_root / "setting").load(setting_name)
        validate_setting_input(setting)

        approximation_data = ExperimentDataLoader(base_dir=self._data_root).load("approximation", approximation_name)
        approximation = _build_approximation_settings(approximation_data)

        source_matrix = (
            matrix_repo.load_matrix(approximation.source_matrix_name)
            if approximation.source_matrix_name is not None
            else matrix
        )
        reference_matrix = (
            matrix_repo.load_matrix(approximation.reference_matrix_name)
            if approximation.reference_matrix_name is not None
            else matrix
        )

        return CompareApproximationFeatureConfig(
            matrix=matrix,
            setting_data=setting,
            approximation=approximation,
            source_matrix=source_matrix,
            reference_matrix=reference_matrix,
        )

    def load_compare_random_approximation(self) -> CompareRandomApproximationFeatureConfig:
        merged = self._config_loader.load_feature_config("compare_random_approximation")
        matrix_name = _require_non_empty_str(merged, key="matrix", name="compare_random_approximation.matrix")
        setting_name = _require_non_empty_str(merged, key="setting", name="compare_random_approximation.setting")
        approximation_name = _require_non_empty_str(
            merged,
            key="approximation",
            name="compare_random_approximation.approximation",
        )
        random_matrix_name = _require_non_empty_str(
            merged,
            key="random_matrix",
            name="compare_random_approximation.random_matrix",
        )

        matrix = MatrixFileInfrastructure(base_dir=self._data_root).load_matrix(matrix_name)
        setting = SettingDataLoader(base_dir=self._data_root / "setting").load(setting_name)
        validate_setting_input(setting)

        exp_loader = ExperimentDataLoader(base_dir=self._data_root)
        approximation_data = exp_loader.load("approximation", approximation_name)
        random_matrix_data = exp_loader.load("random_matrix", random_matrix_name)

        return CompareRandomApproximationFeatureConfig(
            matrix=matrix,
            setting_data=setting,
            approximation=_build_approximation_settings(approximation_data),
            random_matrix=_build_random_matrix_settings(random_matrix_data),
        )


def _build_approximation_settings(data: dict) -> ApproximationSettings:
    return ApproximationSettings(
        source_matrix_name=_optional_non_empty_str(data, key="source_matrix", name="approximation.source_matrix"),
        reference_matrix_name=_optional_non_empty_str(data, key="reference_matrix", name="approximation.reference_matrix"),
        approximation_name=_resolve_algorithm_name(data, kind="approximation", default="MonocycleToGeneralApproximation"),
        distance_name=_resolve_algorithm_name(data, kind="distance", default="MaxElementDifferenceDistance"),
        raw_input=data,
    )


def _build_random_matrix_settings(data: dict) -> RandomMatrixSettings:
    return RandomMatrixSettings(
        size=_required_int(data, key="size", default=None),
        generation_count=_required_int(data, key="generation_count", default=100),
        acceptance_condition=_optional_str(data, key="acceptance_condition", default=""),
        low=_required_float(data, key="low", default=-1.0),
        high=_required_float(data, key="high", default=1.0),
        max_attempts=_required_int(data, key="max_attempts", default=10_000),
        random_seed=_optional_int(data, key="random_seed"),
        raw_input=data,
    )


def _optional_non_empty_str(container: dict, *, key: str, name: str) -> str | None:
    value = container.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} は空でない文字列で指定してください")
    return value


def _require_non_empty_str(container: dict, *, key: str, name: str) -> str:
    value = _optional_non_empty_str(container, key=key, name=name)
    if value is None:
        raise ValueError(f"{name} は必須です")
    return value


def _resolve_algorithm_name(data: dict, *, kind: str, default: str) -> str:
    candidates = [
        data,
        data.get("approximation") if isinstance(data.get("approximation"), dict) else None,
        data.get("approxmation") if isinstance(data.get("approxmation"), dict) else None,
    ]
    for container in candidates:
        if container is None:
            continue
        value = container.get(kind)
        if isinstance(value, str) and value:
            return value
    return default


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


def _optional_str(container: dict, *, key: str, default: str) -> str:
    value = container.get(key, default)
    if not isinstance(value, str):
        raise ValueError(f"{key} は文字列で指定してください")
    return value


def _required_float(config: dict, *, key: str, default: float) -> float:
    raw = config.get(key, default)
    if not isinstance(raw, (int, float)):
        raise ValueError(f"random_matrix.{key} は数値で指定してください")
    return float(raw)
