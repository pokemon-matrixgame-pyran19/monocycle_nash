import numpy as np
from typing import TYPE_CHECKING

from .base import PayoffMatrix
from ..strategy.domain import PureStrategySet

if TYPE_CHECKING:
    from ..equilibrium.domain import MixedStrategy


class GeneralPayoffMatrix(PayoffMatrix):
    """
    一般の利得行列。

    行/列の意味は PureStrategySet で保持する。
    互換性のため labels(list[str]) 指定も受け付ける。
    """

    def __init__(
        self,
        matrix: np.ndarray,
        row_strategies: PureStrategySet | list[str] | None = None,
        col_strategies: PureStrategySet | list[str] | None = None,
    ):
        self._matrix = np.array(matrix, dtype=float)
        if self._matrix.ndim != 2:
            raise ValueError("matrix は2次元配列である必要があります")

        n_row, n_col = self._matrix.shape
        self._row_strategies = self._coerce_strategy_set(
            row_strategies, n_row, id_prefix="s", player_name="row"
        )

        if col_strategies is None:
            if n_row == n_col:
                self._col_strategies = self._row_strategies
            else:
                self._col_strategies = self._coerce_strategy_set(
                    None, n_col, id_prefix="t", player_name="col"
                )
        else:
            self._col_strategies = self._coerce_strategy_set(
                col_strategies, n_col, id_prefix="t", player_name="col"
            )

        if self._matrix.shape != (
            len(self._row_strategies),
            len(self._col_strategies),
        ):
            raise ValueError("行列サイズと戦略数が一致しません")

    @staticmethod
    def _coerce_strategy_set(
        value: PureStrategySet | list[str] | None,
        expected_size: int,
        id_prefix: str,
        player_name: str,
    ) -> PureStrategySet:
        if value is None:
            labels = [f"{id_prefix}{i}" for i in range(expected_size)]
            return PureStrategySet.from_labels(labels, player_name=player_name, id_prefix=id_prefix)
        if isinstance(value, PureStrategySet):
            if len(value) != expected_size:
                raise ValueError("戦略数と行列サイズが一致しません")
            return value
        if len(value) != expected_size:
            raise ValueError("labels の数と行列サイズが一致しません")
        return PureStrategySet.from_labels(value, player_name=player_name, id_prefix=id_prefix)

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

    def solve_equilibrium(self) -> "MixedStrategy":
        """nashpyによる線形最適化で均衡解を計算。"""
        from ..solver.selector import SolverSelector

        selector = SolverSelector()
        return selector.solve(self)
