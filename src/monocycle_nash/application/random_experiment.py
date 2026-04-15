"""Composite UC: ランダム実験 - Generate random matrices and analyze statistics."""

from __future__ import annotations

import itertools
from typing import Any

import numpy as np

from monocycle_nash.domain.character import Character, MatchupVector
from monocycle_nash.domain.matrix.base import PayoffMatrix
from monocycle_nash.domain.matrix.builder import PayoffMatrixBuilder
from monocycle_nash.domain.solver.selector import SolverSelector
from monocycle_nash.domain.team import Team
from monocycle_nash.domain.team_matrix import ExactTeamPayoffCalculator

from .dto import RandomExperimentConfig, RandomExperimentResultDTO
from .single_analysis import SingleAnalysisUseCase


class RandomExperimentUseCase:
    """ランダム実験ユースケース"""

    def __init__(self, analyzer: SingleAnalysisUseCase):
        self._analyzer = analyzer

    def run_team_experiment(
        self, config: RandomExperimentConfig,
    ) -> RandomExperimentResultDTO:
        """チーム行列のランダム実験を実行する。"""
        rng = np.random.default_rng(config.random_seed)

        trials: list[dict[str, Any]] = []
        for i in range(config.generation_count):
            trial = self._run_single_trial(rng, config, trial_index=i)
            trials.append(trial)

        summary, hypothesis_analysis = self._summarize_trials(trials)
        return RandomExperimentResultDTO(
            trials=trials,
            summary={
                **summary,
                "hypothesis_analysis": hypothesis_analysis,
            },
        )

    def _run_single_trial(
        self,
        rng: np.random.Generator,
        config: RandomExperimentConfig,
        *,
        trial_index: int,
    ) -> dict[str, Any]:
        characters = self._generate_random_characters(rng, config)
        character_matrix = PayoffMatrixBuilder.from_characters(characters)

        teams = self._build_pair_teams(character_matrix)
        team_matrix = self._build_team_matrix_strict(character_matrix, teams)

        eq = SolverSelector().solve(team_matrix)

        imag_abs = team_matrix.eigenvalues()
        imag_abs = imag_abs[imag_abs > 1e-10]
        uniq = np.unique(np.round(imag_abs, decimals=10))
        uniq_sorted = np.sort(uniq)[::-1]

        lambda1 = self._float_or_none(uniq_sorted, 0)
        lambda2 = self._float_or_none(uniq_sorted, 1)
        lambda3 = self._float_or_none(uniq_sorted, 2)

        ratio2_to_1 = self._ratio(lambda2, lambda1)
        ratio3_to_1 = self._ratio(lambda3, lambda1)
        dominant_gap = self._ratio(lambda1, lambda2)
        support_size = int(np.sum(eq.probabilities > config.support_threshold))

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

    @staticmethod
    def _generate_random_characters(
        rng: np.random.Generator,
        config: RandomExperimentConfig,
    ) -> list[Character]:
        characters: list[Character] = []
        for i in range(config.character_count):
            characters.append(
                Character(
                    float(rng.uniform(config.power_low, config.power_high)),
                    MatchupVector(
                        float(rng.uniform(config.vector_low, config.vector_high)),
                        float(rng.uniform(config.vector_low, config.vector_high)),
                    ),
                    label=f"c{i}",
                )
            )
        return characters

    @staticmethod
    def _build_pair_teams(character_matrix: PayoffMatrix) -> list[Team]:
        strategies = character_matrix.row_strategies
        teams: list[Team] = []
        for i, j in itertools.combinations(range(len(strategies)), 2):
            left = strategies.get_strategy(i)
            right = strategies.get_strategy(j)
            teams.append(
                Team(
                    label=f"{left.label}+{right.label}",
                    member_ids=(left.id, right.id),
                )
            )
        return teams

    @staticmethod
    def _build_team_matrix_strict(
        character_matrix: PayoffMatrix, teams: list[Team],
    ) -> PayoffMatrix:
        n = len(teams)
        matrix = np.zeros((n, n), dtype=float)
        calculator = ExactTeamPayoffCalculator()
        for i in range(n):
            for j in range(i + 1, n):
                value = calculator.calculate(teams[i], teams[j], character_matrix)
                matrix[i, j] = value
                matrix[j, i] = -value
        return PayoffMatrixBuilder.from_teams(matrix, teams)

    @staticmethod
    def _float_or_none(values: np.ndarray, index: int) -> float | None:
        if values.size <= index:
            return None
        return float(values[index])

    @staticmethod
    def _ratio(
        numerator: float | None, denominator: float | None,
    ) -> float | None:
        if numerator is None or denominator is None or denominator == 0.0:
            return None
        return float(numerator / denominator)

    @staticmethod
    def _summarize_trials(
        trials: list[dict[str, Any]],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        ratio2_values = [
            float(v)
            for v in (t["ratio2_to_1"] for t in trials)
            if v is not None
        ]
        ratio3_values = [
            float(v)
            for v in (t["ratio3_to_1"] for t in trials)
            if v is not None
        ]
        dominant_gap_values = [
            float(v)
            for v in (t["dominant_gap"] for t in trials)
            if v is not None
        ]
        support_sizes = [int(t["support_size"]) for t in trials]

        histogram: dict[str, int] = {}
        for size in support_sizes:
            key = str(size)
            histogram[key] = histogram.get(key, 0) + 1

        # Compute correlation between dominant_gap and support_size
        gaps = np.asarray(
            [float(v) for v in (t["dominant_gap"] for t in trials) if v is not None],
            dtype=float,
        )
        supports_for_gap = np.asarray(
            [int(t["support_size"]) for t in trials if t["dominant_gap"] is not None],
            dtype=float,
        )
        corr: float | None = None
        if gaps.size >= 2 and supports_for_gap.size == gaps.size:
            corr_matrix = np.corrcoef(gaps, supports_for_gap)
            corr_value = float(corr_matrix[0, 1])
            if np.isfinite(corr_value):
                corr = corr_value

        count = len(trials)
        support_size_eq_3_rate = (
            float(sum(1 for s in support_sizes if s == 3) / count)
            if count > 0
            else None
        )
        support_size_le_3_rate = (
            float(sum(1 for s in support_sizes if s <= 3) / count)
            if count > 0
            else None
        )

        ge_2 = RandomExperimentUseCase._support3_rate_for_gap(
            trials, lower=2.0, upper=None,
        )
        lt_2 = RandomExperimentUseCase._support3_rate_for_gap(
            trials, lower=None, upper=2.0,
        )

        summary: dict[str, Any] = {
            "count": count,
            "ratio2_to_1_mean": float(np.mean(ratio2_values)) if ratio2_values else None,
            "ratio2_to_1_std": float(np.std(ratio2_values)) if ratio2_values else None,
            "ratio3_to_1_mean": float(np.mean(ratio3_values)) if ratio3_values else None,
            "ratio3_to_1_std": float(np.std(ratio3_values)) if ratio3_values else None,
            "dominant_gap_mean": (
                float(np.mean(dominant_gap_values)) if dominant_gap_values else None
            ),
            "dominant_gap_std": (
                float(np.std(dominant_gap_values)) if dominant_gap_values else None
            ),
            "support_size_histogram": histogram,
            "support_size_eq_3_rate": support_size_eq_3_rate,
            "support_size_le_3_rate": support_size_le_3_rate,
            "support_size_gt_3_rate": (
                (1.0 - support_size_le_3_rate) if support_size_le_3_rate is not None else None
            ),
            "corr_dominant_gap_vs_support_size": corr,
            "support3_rate_gap_ge_2": ge_2,
            "support3_rate_gap_lt_2": lt_2,
            "delta_support3_rate": (
                ge_2 - lt_2 if ge_2 is not None and lt_2 is not None else None
            ),
        }

        hypothesis_analysis = RandomExperimentUseCase._build_hypothesis_analysis(
            trials,
        )
        return summary, hypothesis_analysis

    @staticmethod
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
        return float(
            sum(1 for t in filtered if bool(t["is_support_3"])) / len(filtered)
        )

    @staticmethod
    def _build_hypothesis_analysis(
        trials: list[dict[str, Any]],
    ) -> dict[str, Any]:
        gap_bins: list[tuple[float, float | None, str]] = [
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
                if float(t["dominant_gap"]) >= lower
                and (upper is None or float(t["dominant_gap"]) < upper)
            ]
            count = len(bucket)
            support3_rate = (
                float(sum(1 for t in bucket if bool(t["is_support_3"])) / count)
                if count > 0
                else None
            )
            mean_support_size = (
                float(np.mean([int(t["support_size"]) for t in bucket]))
                if count > 0
                else None
            )
            bin_results.append({
                "bin": label,
                "count": count,
                "support3_rate": support3_rate,
                "mean_support_size": mean_support_size,
            })

        overall_support3_rate = (
            float(
                sum(1 for t in defined_trials if bool(t["is_support_3"]))
                / len(defined_trials)
            )
            if defined_trials
            else None
        )

        return {
            "gap_bins": bin_results,
            "overall_support3_rate": overall_support3_rate,
        }
