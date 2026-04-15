"""Data transfer objects used by application layer use cases."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from monocycle_nash.domain.equilibrium import MixedStrategy
from monocycle_nash.domain.matrix.base import PayoffMatrix


@dataclass(frozen=True)
class MatrixInputDTO:
    """行列構築の入力DTO"""

    # General matrix input
    raw_matrix: list[list[float]] | None = None
    labels: list[str] | None = None
    # Character input
    characters: list[dict] | None = None  # [{"label": "A", "p": 1.0, "v": [0.5, 0.3]}, ...]
    # Team input
    team_mode: str | None = None  # "strict" | "2by2" | "monocycle" | None
    teams: list[dict] | None = None  # [{"label": "T1", "members": [0, 1]}, ...]


@dataclass(frozen=True)
class EquilibriumResultDTO:
    """均衡解の結果DTO"""

    mixed_strategy: MixedStrategy
    pure_payoffs: list[float]
    divergence: list[float]
    eigenvalues: list[float] | None = None


@dataclass(frozen=True)
class AnalysisConfig:
    """分析設定"""

    solve_equilibrium: bool = True
    generate_payoff_graph: bool = False
    generate_character_plot: bool = False
    graph_threshold: float = 0.0
    graph_canvas_size: int = 840
    character_canvas_size: int = 840
    character_margin: int = 90


@dataclass(frozen=True)
class AnalysisResultDTO:
    """単体分析結果DTO"""

    matrix: PayoffMatrix
    equilibrium: EquilibriumResultDTO | None = None
    payoff_graph_svg: str | None = None
    character_plot_svg: str | None = None


@dataclass(frozen=True)
class ComparisonResultDTO:
    """比較結果DTO"""

    source_analysis: AnalysisResultDTO
    reference_analysis: AnalysisResultDTO
    max_element_distance: float | None = None
    equilibrium_distance: float | None = None
    approximation_quality: float | None = None


@dataclass(frozen=True)
class RandomExperimentConfig:
    """ランダム実験設定"""

    character_count: int = 6
    team_size: int = 2
    generation_count: int = 100
    random_seed: int | None = None
    power_low: float = -1.0
    power_high: float = 1.0
    vector_low: float = -1.0
    vector_high: float = 1.0
    support_threshold: float = 1e-6
    use_monocycle_formula: bool = True


@dataclass
class RandomExperimentResultDTO:
    """ランダム実験結果DTO"""

    trials: list[dict] = field(default_factory=list)
    summary: dict = field(default_factory=dict)
