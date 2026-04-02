from __future__ import annotations

from abc import ABC, abstractmethod
import csv
from dataclasses import asdict, dataclass
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from monocycle_nash.equilibrium.infra.solver.selector import SolverSelector
from monocycle_nash.game.infra.matrix import build_matrix_from_input
from monocycle_nash.runtime.infra.loader.main_config import MainConfigLoader
from monocycle_nash.runtime.infra.loader.runtime_common import (
    _to_toml,
    prepare_run_session,
    write_input_snapshots,
    write_json,
)
from monocycle_nash.runtime.infra.runmeta.setting_domain import RuntimeSetting


FEATURE_NAME = "experiment_team_strict_spectrum"


@dataclass(frozen=True)
class TeamStrictExperimentSettings:
    character_count: int
    team_size: int
    generation_count: int
    random_seed: int | None
    power_low: float
    power_high: float
    vector_low: float
    vector_high: float
    support_threshold: float = 1e-6
    solver: str = ""


@dataclass(frozen=True)
class TeamStrictSpectrumFeatureConfig:
    setting_data: RuntimeSetting
    experiment: TeamStrictExperimentSettings


class TeamStrictSpectrumSettingLoader(ABC):
    @abstractmethod
    def load_experiment_team_strict_spectrum(self) -> TeamStrictSpectrumFeatureConfig:
        raise NotImplementedError


def run(config_loader: MainConfigLoader) -> int:
    from monocycle_nash.analysis.infra.approximation import ApproximationFeatureInfrastructure

    setting_loader: TeamStrictSpectrumSettingLoader = ApproximationFeatureInfrastructure(config_loader)
    feature_config = setting_loader.load_experiment_team_strict_spectrum()
    settings = feature_config.experiment

    service, ctx, conn = prepare_run_session(feature_config.setting_data, f"uv run main ({FEATURE_NAME})")
    try:
        snapshot_matrix_data = _build_matrix_input(np.random.default_rng(settings.random_seed), settings)
        write_input_snapshots(
            service,
            ctx.run_id,
            matrix_data=snapshot_matrix_data,
            graph_data=None,
            setting_data=feature_config.setting_data,
        )
        input_dir = service.artifact_store.run_dir(ctx.run_id) / "input"
        (input_dir / "team_strict_spectrum.toml").write_text(
            _to_toml(_experiment_to_toml_payload(settings)),
            encoding="utf-8",
        )

        rng = np.random.default_rng(settings.random_seed)
        trials = [_run_single_trial(rng, settings, trial_index=i) for i in range(settings.generation_count)]
        summary, hypothesis_analysis = _summarize_trials(trials)
        output_dir = service.artifact_store.run_dir(ctx.run_id) / "output"
        _write_trials_csv(output_dir / "team_strict_spectrum_trials.csv", trials)

        write_json(
            output_dir / "team_strict_spectrum_experiment.json",
            {
                "feature": FEATURE_NAME,
                "settings": asdict(settings),
                "trials": trials,
                "summary": summary,
                "hypothesis_analysis": hypothesis_analysis,
            },
        )

        service.finish_success(
            ctx,
            extra_meta={
                "output_files": [
                    "output/team_strict_spectrum_experiment.json",
                    "output/team_strict_spectrum_trials.csv",
                ]
            },
        )
        return 0
    except Exception as exc:  # noqa: BLE001
        err = traceback.format_exc()
        (service.artifact_store.run_dir(ctx.run_id) / "logs" / "stderr.log").write_text(err, encoding="utf-8")
        service.finish_fail(ctx, extra_meta={"error": str(exc)})
        return 1
    finally:
        conn.close()


def _run_single_trial(
    rng: np.random.Generator,
    settings: TeamStrictExperimentSettings,
    *,
    trial_index: int,
) -> dict[str, Any]:
    matrix_data = _build_matrix_input(rng, settings)
    matrix = build_matrix_from_input(matrix_data)
    eq = SolverSelector().solve(matrix)

    imag_abs = matrix.eigenvalues()
    imag_abs = imag_abs[imag_abs > 1e-10]
    uniq = np.unique(np.round(imag_abs, decimals=10))
    uniq_sorted = np.sort(uniq)[::-1]

    lambda1 = _float_or_none(uniq_sorted, 0)
    lambda2 = _float_or_none(uniq_sorted, 1)
    lambda3 = _float_or_none(uniq_sorted, 2)

    ratio2_to_1 = _ratio(lambda2, lambda1)
    ratio3_to_1 = _ratio(lambda3, lambda1)
    dominant_gap = _ratio(lambda1, lambda2)
    support_size = int(np.sum(eq.probabilities > settings.support_threshold))

    return {
        "trial_index": trial_index,
        "lambda1": lambda1,
        "lambda2": lambda2,
        "lambda3": lambda3,
        "ratio2_to_1": ratio2_to_1,
        "ratio3_to_1": ratio3_to_1,
        "dominant_gap": dominant_gap,
        "support_size": support_size,
        "is_support_3": support_size == 3,
    }


def _build_matrix_input(rng: np.random.Generator, settings: TeamStrictExperimentSettings) -> dict[str, Any]:
    if settings.character_count != 6:
        raise ValueError("experiment_team_strict_spectrum.character_count は 6 固定です")
    if settings.team_size != 2:
        raise ValueError("experiment_team_strict_spectrum.team_size は 2 固定です")

    characters: list[dict[str, Any]] = []
    for i in range(settings.character_count):
        characters.append(
            {
                "label": f"c{i}",
                "p": float(rng.uniform(settings.power_low, settings.power_high)),
                "v": [
                    float(rng.uniform(settings.vector_low, settings.vector_high)),
                    float(rng.uniform(settings.vector_low, settings.vector_high)),
                ],
            }
        )
    return {
        "characters": characters,
        "team": "strict",
    }


def _experiment_to_toml_payload(settings: TeamStrictExperimentSettings) -> dict[str, Any]:
    payload = asdict(settings)
    if payload["solver"] == "":
        del payload["solver"]
    return payload


def _float_or_none(values: np.ndarray, index: int) -> float | None:
    if values.size <= index:
        return None
    return float(values[index])


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator == 0.0:
        return None
    return float(numerator / denominator)


def _summarize_trials(trials: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    ratio2_values = [float(v) for v in (t["ratio2_to_1"] for t in trials) if v is not None]
    ratio3_values = [float(v) for v in (t["ratio3_to_1"] for t in trials) if v is not None]
    dominant_gap_values = [float(v) for v in (t["dominant_gap"] for t in trials) if v is not None]
    support_sizes = [int(t["support_size"]) for t in trials]

    histogram: dict[str, int] = {}
    for size in support_sizes:
        key = str(size)
        histogram[key] = histogram.get(key, 0) + 1

    gaps = np.asarray([float(v) for v in (t["dominant_gap"] for t in trials) if v is not None], dtype=float)
    supports_for_gap = np.asarray(
        [
            int(t["support_size"])
            for t in trials
            if t["dominant_gap"] is not None
        ],
        dtype=float,
    )
    corr = None
    if gaps.size >= 2 and supports_for_gap.size == gaps.size:
        corr_matrix = np.corrcoef(gaps, supports_for_gap)
        corr_value = float(corr_matrix[0, 1])
        if np.isfinite(corr_value):
            corr = corr_value

    count = len(trials)
    support_size_eq_3_rate = (
        float(sum(1 for size in support_sizes if size == 3) / count)
        if count > 0
        else None
    )

    summary = {
        "count": count,
        "ratio2_to_1_mean": float(np.mean(ratio2_values)) if ratio2_values else None,
        "ratio2_to_1_std": float(np.std(ratio2_values)) if ratio2_values else None,
        "ratio3_to_1_mean": float(np.mean(ratio3_values)) if ratio3_values else None,
        "ratio3_to_1_std": float(np.std(ratio3_values)) if ratio3_values else None,
        "dominant_gap_mean": float(np.mean(dominant_gap_values)) if dominant_gap_values else None,
        "dominant_gap_std": float(np.std(dominant_gap_values)) if dominant_gap_values else None,
        "support_size_histogram": histogram,
        "support_size_eq_3_rate": support_size_eq_3_rate,
        "corr_dominant_gap_vs_support_size": corr,
        "support3_rate_gap_ge_2": _support3_rate_for_gap(trials, lower=2.0, upper=None),
        "support3_rate_gap_lt_2": _support3_rate_for_gap(trials, lower=None, upper=2.0),
    }
    ge_2 = summary["support3_rate_gap_ge_2"]
    lt_2 = summary["support3_rate_gap_lt_2"]
    summary["delta_support3_rate"] = ge_2 - lt_2 if ge_2 is not None and lt_2 is not None else None

    hypothesis_analysis = _build_hypothesis_analysis(trials)
    return summary, hypothesis_analysis


def _support3_rate_for_gap(
    trials: list[dict[str, Any]],
    *,
    lower: float | None,
    upper: float | None,
) -> float | None:
    filtered = [
        t
        for t in trials
        if t["dominant_gap"] is not None
        and (lower is None or float(t["dominant_gap"]) >= lower)
        and (upper is None or float(t["dominant_gap"]) < upper)
    ]
    if not filtered:
        return None
    return float(sum(1 for t in filtered if bool(t["is_support_3"])) / len(filtered))


def _build_hypothesis_analysis(trials: list[dict[str, Any]]) -> dict[str, Any]:
    gap_bins = [
        (1.0, 1.2, "[1.0,1.2)"),
        (1.2, 1.5, "[1.2,1.5)"),
        (1.5, 2.0, "[1.5,2.0)"),
        (2.0, 3.0, "[2.0,3.0)"),
        (3.0, None, "[3.0,inf)"),
    ]

    bin_results: list[dict[str, Any]] = []
    defined_trials = [t for t in trials if t["dominant_gap"] is not None]
    for lower, upper, label in gap_bins:
        bucket = [
            t
            for t in defined_trials
            if float(t["dominant_gap"]) >= lower and (upper is None or float(t["dominant_gap"]) < upper)
        ]
        count = len(bucket)
        support3_rate = float(sum(1 for t in bucket if bool(t["is_support_3"])) / count) if count > 0 else None
        mean_support_size = float(np.mean([int(t["support_size"]) for t in bucket])) if count > 0 else None
        bin_results.append(
            {
                "bin": label,
                "count": count,
                "support3_rate": support3_rate,
                "mean_support_size": mean_support_size,
            }
        )

    overall_support3_rate = (
        float(sum(1 for t in defined_trials if bool(t["is_support_3"])) / len(defined_trials))
        if defined_trials
        else None
    )

    return {
        "gap_bins": bin_results,
        "overall_support3_rate": overall_support3_rate,
    }


def _write_trials_csv(path: Path, trials: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "trial",
        "lambda1",
        "lambda2",
        "lambda3",
        "ratio2_to_1",
        "ratio3_to_1",
        "dominant_gap",
        "support_size",
        "is_support_3",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for t in trials:
            writer.writerow(
                {
                    "trial": t["trial_index"],
                    "lambda1": t["lambda1"],
                    "lambda2": t["lambda2"],
                    "lambda3": t["lambda3"],
                    "ratio2_to_1": t["ratio2_to_1"],
                    "ratio3_to_1": t["ratio3_to_1"],
                    "dominant_gap": t["dominant_gap"],
                    "support_size": t["support_size"],
                    "is_support_3": t["is_support_3"],
                }
            )
