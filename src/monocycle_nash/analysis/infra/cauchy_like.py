from __future__ import annotations

from monocycle_nash.analysis.app.compare_product_formula import (
    CauchyLikeSettings,
    CompareProductFormulaFeatureConfig,
    CompareProductFormulaSettingLoader,
)
from monocycle_nash.runtime.infra.loader.data_loader import ExperimentDataLoader, SettingDataLoader
from monocycle_nash.runtime.infra.loader.main_config import MainConfigLoader
from monocycle_nash.runtime.infra.loader.runtime_common import TomlRuntimeSettingParser


class CauchyLikeFeatureInfrastructure(CompareProductFormulaSettingLoader):
    def __init__(self, config_loader: MainConfigLoader):
        self._config_loader = config_loader
        self._data_root = config_loader.data_root

    def load_compare_product_formula(self) -> CompareProductFormulaFeatureConfig:
        merged = self._config_loader.load_feature_config("compare_product_formula")
        cauchy_like_name = _require_non_empty_str(
            merged, key="cauchy_like", name="compare_product_formula.cauchy_like"
        )
        setting_name = _require_non_empty_str(
            merged, key="setting", name="compare_product_formula.setting"
        )

        cauchy_like_data = ExperimentDataLoader(base_dir=self._data_root).load(
            "cauchy_like", cauchy_like_name
        )
        cl_settings = _build_cauchy_like_settings(cauchy_like_data)

        setting = TomlRuntimeSettingParser().parse(
            SettingDataLoader(base_dir=self._data_root / "setting").load(setting_name)
        )

        return CompareProductFormulaFeatureConfig(
            setting_data=setting,
            cauchy_like=cl_settings,
        )


def _build_cauchy_like_settings(data: dict) -> CauchyLikeSettings:
    a = _required_float_list(data, key="a", name="cauchy_like.a")
    b = _required_float_list(data, key="b", name="cauchy_like.b")
    if len(a) != len(b):
        raise ValueError("cauchy_like.a と cauchy_like.b の長さが一致しません")
    labels_raw = data.get("labels")
    if labels_raw is not None:
        if not isinstance(labels_raw, list) or any(not isinstance(x, str) for x in labels_raw):
            raise ValueError("cauchy_like.labels は文字列配列で指定してください")
        if len(labels_raw) != len(a):
            raise ValueError("cauchy_like.labels の長さが a と一致しません")
        labels: list[str] | None = [str(x) for x in labels_raw]
    else:
        labels = None
    return CauchyLikeSettings(a=a, b=b, labels=labels)


def _required_float_list(data: dict, *, key: str, name: str) -> list[float]:
    raw = data.get(key)
    if raw is None:
        raise ValueError(f"{name} は必須です")
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{name} は空でない配列で指定してください")
    if any(not isinstance(x, (int, float)) for x in raw):
        raise ValueError(f"{name} は数値配列で指定してください")
    return [float(x) for x in raw]


def _require_non_empty_str(container: dict, *, key: str, name: str) -> str:
    value = container.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} は必須です")
    return value
