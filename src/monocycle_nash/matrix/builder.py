import numpy as np
from typing import TYPE_CHECKING

from .general import GeneralPayoffMatrix
from .monocycle import MonocyclePayoffMatrix
from .random import (
    RandomMatrixAcceptanceCondition,
    generate_random_skew_symmetric_matrix,
)
from ..character.domain import Character
from ..strategy.domain import PureStrategySet
from ..team.matrix_approx import TwoPlayerTeamMatrixCalculator

if TYPE_CHECKING:
    from ..matrix.base import PayoffMatrix
    from ..team.domain import Team


class PayoffMatrixBuilder:
    """
    利得行列ビルダー。

    Character/Team を PureStrategySet に正規化してから行列を構築する。
    """

    @staticmethod
    def from_characters(
        characters: list[Character],
        labels: list[str] | None = None,
    ) -> MonocyclePayoffMatrix:
        """Characterリストから単相性モデル利得行列を生成。"""
        return MonocyclePayoffMatrix(characters, labels=labels)

    @staticmethod
    def from_general_matrix(
        matrix: np.ndarray,
        labels: list[str] | None = None,
        row_strategies: PureStrategySet | None = None,
        col_strategies: PureStrategySet | None = None,
    ) -> GeneralPayoffMatrix:
        """
        任意行列から一般利得行列を生成。

        `row_strategies` が指定されればそれを優先し、未指定時は labels/default を使用。
        """
        if row_strategies is not None:
            return GeneralPayoffMatrix(matrix, row_strategies, col_strategies)
        return GeneralPayoffMatrix(matrix, labels, col_strategies)

    @staticmethod
    def from_teams(team_payoff: np.ndarray, teams: list["Team"]) -> GeneralPayoffMatrix:
        """既に計算済みのTeam利得行列から一般利得行列を生成。"""
        row_strategies = PureStrategySet.from_teams(teams, player_name="row")
        return GeneralPayoffMatrix(team_payoff, row_strategies, row_strategies)

    @staticmethod
    def from_team_matchups(
        teams: list["Team"],
        character_matrix: "PayoffMatrix",
        use_monocycle_formula: bool = True,
    ) -> GeneralPayoffMatrix:
        """Character利得行列からTeam利得行列を生成。"""
        matrix_calculator = TwoPlayerTeamMatrixCalculator(
            character_matrix=character_matrix,
            use_monocycle_formula=use_monocycle_formula,
        )
        return matrix_calculator.generate_matrix(teams)

    @staticmethod
    def from_random_matrix(
        size: int,
        low: float = -1.0,
        high: float = 1.0,
        acceptance_condition: RandomMatrixAcceptanceCondition | None = None,
        *,
        rng: np.random.Generator | None = None,
        max_attempts: int = 10_000,
        labels: list[str] | None = None,
    ) -> GeneralPayoffMatrix:
        """交代行列のランダム利得行列を生成する。"""
        matrix = generate_random_skew_symmetric_matrix(
            size=size,
            low=low,
            high=high,
            acceptance_condition=acceptance_condition,
            rng=rng,
            max_attempts=max_attempts,
        )
        return GeneralPayoffMatrix(matrix, labels)
