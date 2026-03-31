from __future__ import annotations

from pathlib import Path

from monocycle_nash.approximation.compare_approximation_app import (
    ApproximationSettings,
    CompareApproximationFeatureConfig,
    CompareApproximationSettingLoader,
)
from monocycle_nash.approximation.compare_random_approximation_app import (
    CompareRandomApproximationFeatureConfig,
    CompareRandomApproximationSettingLoader,
    RandomMatrixSettings,
)
from monocycle_nash.loader.data_loader import ExperimentDataLoader, SettingDataLoader
from monocycle_nash.loader.runtime_common import TomlRuntimeSettingParser
from monocycle_nash.loader.toml_tree import TomlTreeLoader
from monocycle_nash.matrix import MatrixFileInfrastructure


class ApproximationFeatureInfrastructure(CompareApproximationSettingLoader, CompareRandomApproximationSettingLoader):
    def __init__(self, feature_config_path: Path | str):
        self._config_path = Path(feature_config_path)
        self._data_root = self._config_path.parent.parent
        self._tree_loader = TomlTreeLoader()

    def load_compare_approximation(self) -> CompareApproximationFeatureConfig:
        merged = self._load_feature_config()
        matrix_name = _require_non_empty_str(merged, key="matrix", name="compare_approximation.matrix")
        setting_name = _require_non_empty_str(merged, key="setting", name="compare_approximation.setting")
        approximation_name = _require_non_empty_str(merged, key="approximation", name="compare_approximation.approximation")

        matrix_repo = MatrixFileInfrastructure(base_dir=self._data_root)
        matrix = matrix_repo.load_matrix(matrix_name)
        setting = TomlRuntimeSettingParser().parse(
            SettingDataLoader(base_dir=self._data_root / "setting").load(setting_name)
        )

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
        merged = self._load_feature_config()
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
        setting = TomlRuntimeSettingParser().parse(
            SettingDataLoader(base_dir=self._data_root / "setting").load(setting_name)
        )

        exp_loader = ExperimentDataLoader(base_dir=self._data_root)
        approximation_data = exp_loader.load("approximation", approximation_name)
        random_matrix_data = exp_loader.load("random_matrix", random_matrix_name)

        return CompareRandomApproximationFeatureConfig(
            matrix=matrix,
            setting_data=setting,
            approximation=_build_approximation_settings(approximation_data),
            random_matrix=_build_random_matrix_settings(random_matrix_data),
        )

    def _load_feature_config(self) -> dict:
        return self._tree_loader.load(self._config_path)


def _build_approximation_settings(data: dict) -> ApproximationSettings:
    return ApproximationSettings(
        source_matrix_name=_optional_non_empty_str(data, key="source_matrix", name="approximation.source_matrix"),
        reference_matrix_name=_optional_non_empty_str(data, key="reference_matrix", name="approximation.reference_matrix"),
        approximation_name=_resolve_algorithm_name(data, kind="approximation", default="MonocycleToGeneralApproximation"),
        distance_name=_resolve_algorithm_name(data, kind="distance", default="MaxElementDifferenceDistance"),
        dominant_eigen_ratio_bin_edges=_optional_float_tuple(data, key="dominant_eigen_ratio_bin_edges"),
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
    )


def _optional_float_tuple(container: dict, *, key: str) -> tuple[float, ...] | None:
    value = container.get(key)
    if value is None:
        return None
    if not isinstance(value, list) or any(not isinstance(x, (int, float)) for x in value):
        raise ValueError(f"{key} は数値配列で指定してください")
    edges = tuple(float(x) for x in value)
    if any(x <= 0.0 for x in edges):
        raise ValueError(f"{key} は正の数値配列で指定してください")
    if any(edges[i] >= edges[i + 1] for i in range(len(edges) - 1)):
        raise ValueError(f"{key} は昇順で指定してください")
    return edges


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
