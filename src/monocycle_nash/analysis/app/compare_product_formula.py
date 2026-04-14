"""
product_formula と厳密均衡解の比較ユースケース

CauchyLike 行列に対し、
  - product_formula(): v_i = Π_{j≠i}(b_i - b_j)
  - a 除算・正規化した近似値 (v_i / a_i を確率化)
  - theoretical_equilibrium() == solve_equilibrium() による厳密解
を並べて比較し、近似が有効かどうか（特に n≥5 で）を確認する。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import traceback

import numpy as np

from monocycle_nash.runtime.infra.loader.main_config import MainConfigLoader
from monocycle_nash.runtime.infra.loader.runtime_common import (
    _to_toml,
    prepare_run_session,
    write_json,
)
from monocycle_nash.game.domain.matrix.cauchy_like import CauchyLikePayoffMatrix
from monocycle_nash.runtime.infra.runmeta.setting_domain import RuntimeSetting


FEATURE_NAME = "compare_product_formula"


@dataclass(frozen=True)
class CauchyLikeSettings:
    a: list[float]
    b: list[float]
    labels: list[str] | None


@dataclass(frozen=True)
class CompareProductFormulaFeatureConfig:
    setting_data: RuntimeSetting
    cauchy_like: CauchyLikeSettings


class CompareProductFormulaSettingLoader(ABC):
    @abstractmethod
    def load_compare_product_formula(self) -> CompareProductFormulaFeatureConfig:
        raise NotImplementedError


def run(config_loader: MainConfigLoader) -> int:
    from monocycle_nash.analysis.infra.cauchy_like import CauchyLikeFeatureInfrastructure

    try:
        setting_loader: CompareProductFormulaSettingLoader = CauchyLikeFeatureInfrastructure(config_loader)
        feature_config = setting_loader.load_compare_product_formula()
    except Exception:  # noqa: BLE001
        return 1

    cl_settings = feature_config.cauchy_like

    service, ctx, conn = prepare_run_session(feature_config.setting_data, f"uv run main ({FEATURE_NAME})")
    try:
        matrix = CauchyLikePayoffMatrix.from_ab_lists(
            cl_settings.a,
            cl_settings.b,
            labels=cl_settings.labels,
        )

        input_dir = service.artifact_store.run_dir(ctx.run_id) / "input"
        input_dir.mkdir(parents=True, exist_ok=True)
        (input_dir / "cauchy_like.toml").write_text(
            _to_toml(_cauchy_like_to_toml_payload(cl_settings)),
            encoding="utf-8",
        )

        exact = matrix.theoretical_equilibrium()
        product_raw = matrix.product_formula()
        a = matrix.get_a_values()

        naive_weights = product_raw / a
        naive_valid, naive_probs = _try_normalize_positive(naive_weights)

        residual_exact = matrix.matrix @ exact
        max_residual_exact = float(np.max(np.abs(residual_exact)))

        output: dict = {
            "n": matrix.size,
            "a": a.tolist(),
            "b": matrix.get_b_values().tolist(),
            "exact_equilibrium": {
                "probabilities": exact.tolist(),
                "max_residual": max_residual_exact,
            },
            "product_formula_raw": product_raw.tolist(),
            "product_formula_normalized": {
                "is_valid": naive_valid,
                "probabilities": naive_probs.tolist() if naive_probs is not None else None,
                "max_residual": float(np.max(np.abs(matrix.matrix @ naive_probs)))
                if naive_probs is not None
                else None,
                "max_diff_from_exact": float(np.max(np.abs(naive_probs - exact)))
                if naive_probs is not None
                else None,
            },
        }

        write_json(
            service.artifact_store.run_dir(ctx.run_id) / "output" / "compare_product_formula.json",
            output,
        )

        service.finish_success(ctx, extra_meta={"output_files": ["output/compare_product_formula.json"]})
        return 0
    except Exception as exc:  # noqa: BLE001
        err = traceback.format_exc()
        (service.artifact_store.run_dir(ctx.run_id) / "logs" / "stderr.log").write_text(err, encoding="utf-8")
        service.finish_fail(ctx, extra_meta={"error": str(exc)})
        return 1
    finally:
        conn.close()


def _try_normalize_positive(
    weights: np.ndarray,
) -> tuple[bool, np.ndarray | None]:
    """全成分が非負かどうかを確認し、正規化した確率を返す。"""
    w = np.asarray(weights, dtype=float)
    if np.all(w <= 0):
        w = -w
    if np.any(w < -1e-10):
        return False, None
    w = np.maximum(w, 0.0)
    total = w.sum()
    if total < 1e-15:
        return False, None
    return True, w / total


def _cauchy_like_to_toml_payload(settings: CauchyLikeSettings) -> dict:
    payload: dict = {"a": settings.a, "b": settings.b}
    if settings.labels is not None:
        payload["labels"] = settings.labels
    return payload
