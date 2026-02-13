import numpy as np
from typing import TYPE_CHECKING

from .base import PayoffMatrix
from ..character.domain import Character, MatchupVector
from ..strategy.domain import MonocyclePureStrategy, PureStrategySet

if TYPE_CHECKING:
    from ..equilibrium.domain import MixedStrategy


class MonocyclePayoffMatrix(PayoffMatrix):
    """
    単相性モデルの利得行列。

    Aij = pi - pj + vi×vj を満たし、
    行/列の意味は MonocyclePureStrategy で保持する。
    """

    def __init__(
        self,
        characters_or_row_strategies: list[Character] | PureStrategySet,
        labels: list[str] | None = None,
        col_strategies: list[Character] | PureStrategySet | None = None,
    ):
        if isinstance(characters_or_row_strategies, PureStrategySet):
            if labels is not None:
                raise ValueError("row_strategies指定時は labels を同時に指定できません")
            self._row_strategies = characters_or_row_strategies
        else:
            self._row_strategies = PureStrategySet.from_characters(
                characters_or_row_strategies,
                player_name="row",
                ids=labels,
                id_prefix="c",
            )

        if col_strategies is None:
            self._col_strategies = self._row_strategies
        elif isinstance(col_strategies, PureStrategySet):
            self._col_strategies = col_strategies
        else:
            self._col_strategies = PureStrategySet.from_characters(
                col_strategies,
                player_name="col",
                id_prefix="d",
            )

        self._validate_monocycle_set(self._row_strategies)
        self._validate_monocycle_set(self._col_strategies)

        self._matrix = self._calculate_matrix()

    @classmethod
    def from_characters(
        cls,
        characters: list[Character],
        labels: list[str] | None = None,
    ) -> "MonocyclePayoffMatrix":
        return cls(characters, labels=labels)

    @staticmethod
    def _validate_monocycle_set(strategies: PureStrategySet) -> None:
        for strategy in strategies:
            MonocyclePureStrategy.cast(strategy)

    @staticmethod
    def _to_monocycle_strategies(strategies: PureStrategySet) -> list[MonocyclePureStrategy]:
        return [MonocyclePureStrategy.cast(strategy) for strategy in strategies]

    @property
    def matrix(self) -> np.ndarray:
        return self._matrix

    @property
    def size(self) -> int:
        return len(self._row_strategies)

    @property
    def row_strategies(self) -> PureStrategySet:
        return self._row_strategies

    @property
    def col_strategies(self) -> PureStrategySet:
        return self._col_strategies

    @property
    def labels(self) -> list[str]:
        """後方互換: 行プレイヤー側ラベル。"""
        return self._row_strategies.labels

    @property
    def characters(self) -> list[Character]:
        """後方互換: 行プレイヤー側 Character リスト。"""
        return [strategy.entity for strategy in self._to_monocycle_strategies(self._row_strategies)]

    def _calculate_matrix(self) -> np.ndarray:
        row_strategies = self._to_monocycle_strategies(self._row_strategies)
        col_strategies = self._to_monocycle_strategies(self._col_strategies)
        n_row = len(row_strategies)
        n_col = len(col_strategies)
        matrix = np.zeros((n_row, n_col))

        for i, row in enumerate(row_strategies):
            for j, col in enumerate(col_strategies):
                matrix[i, j] = row.power - col.power + row.vector.times(col.vector)
        return matrix

    def solve_equilibrium(self) -> "MixedStrategy":
        from ..solver.selector import SolverSelector

        selector = SolverSelector()
        return selector.solve(self)

    def shift_origin(self, a_vector: MatchupVector) -> "MonocyclePayoffMatrix":
        shifted_rows = self._row_strategies.shift_origin(a_vector)
        shifted_cols = self._col_strategies.shift_origin(a_vector)
        return MonocyclePayoffMatrix(shifted_rows, col_strategies=shifted_cols)

    def get_power_vector(self) -> np.ndarray:
        return np.array([strategy.power for strategy in self._to_monocycle_strategies(self._row_strategies)])

    def get_matchup_vectors(self) -> np.ndarray:
        return np.array(
            [[strategy.vector.x, strategy.vector.y] for strategy in self._to_monocycle_strategies(self._row_strategies)]
        )
