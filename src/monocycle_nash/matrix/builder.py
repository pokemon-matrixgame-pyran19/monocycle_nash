import numpy as np
from typing import TYPE_CHECKING

from .general import GeneralPayoffMatrix
from .monocycle import MonocyclePayoffMatrix
from ..character.domain import Character
from ..strategy.domain import PureStrategySet

if TYPE_CHECKING:
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
        """Teamリストから一般利得行列を生成。"""
        row_strategies = PureStrategySet.from_teams(teams, player_name="row")
        return GeneralPayoffMatrix(team_payoff, row_strategies, row_strategies)
