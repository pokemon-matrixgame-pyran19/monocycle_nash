"""
第二のコーシー型利得行列 (Cauchy-like XY payoff matrix)

行列の形: B_ij = 1 / (x_i * y_j - x_j * y_i)  (i ≠ j), B_ii = 0
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from monocycle_nash.domain.matrix.base import PayoffMatrix
from monocycle_nash.domain.strategy import PureStrategy, PureStrategySet


@dataclass(frozen=True)
class CauchyLikeXYEntity:
    """
    第二のコーシー型行列の行/列に対応するエンティティ。

    Attributes:
        x: パラメータ x_i
        y: パラメータ y_i
        label: 表示用ラベル
    """

    x: float
    y: float
    label: str = ""


@dataclass(frozen=True)
class CauchyLikeXYPureStrategy(PureStrategy):
    """第二のコーシー型行列専用の純粋戦略。"""

    entity: CauchyLikeXYEntity

    @property
    def x(self) -> float:
        return self.entity.x

    @property
    def y(self) -> float:
        return self.entity.y

    @staticmethod
    def cast(strategy: PureStrategy) -> "CauchyLikeXYPureStrategy":
        if not isinstance(strategy, CauchyLikeXYPureStrategy):
            raise TypeError("CauchyLikeXYPureStrategy が必要です")
        return strategy


class CauchyLikeXYPayoffMatrix(PayoffMatrix):
    """
    第二のコーシー型利得行列。

    行列の要素は以下で定義される:
        B_ij = 1 / (x_i * y_j - x_j * y_i)  (i ≠ j)
        B_ii = 0
    """

    def __init__(
        self,
        params: list[tuple[float, float]] | list[tuple[float, float, str]],
        labels: list[str] | None = None,
    ) -> None:
        strategies: list[PureStrategy] = []
        for i, p in enumerate(params):
            if len(p) == 3:
                x, y, lbl = p[0], p[1], p[2]
            else:
                x, y = p[0], p[1]
                lbl = labels[i] if labels is not None else f"s{i}"
            entity = CauchyLikeXYEntity(x=float(x), y=float(y), label=lbl)
            strategies.append(CauchyLikeXYPureStrategy(id=lbl, entity=entity))

        self._row_strategies = PureStrategySet(strategies=strategies, player_name="row")
        self._col_strategies = self._row_strategies
        self._matrix = self._calculate_matrix()

    @classmethod
    def from_xy_lists(
        cls,
        x_list: list[float],
        y_list: list[float],
        labels: list[str] | None = None,
    ) -> "CauchyLikeXYPayoffMatrix":
        if len(x_list) != len(y_list):
            raise ValueError("x_list と y_list の長さが一致しません")
        if labels is not None and len(labels) != len(x_list):
            raise ValueError("labels の長さが x_list と一致しません")
        params: list[tuple[float, float]] = list(zip(x_list, y_list))
        return cls(params, labels=labels)

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

    def _calculate_matrix(self) -> np.ndarray:
        strategies = [CauchyLikeXYPureStrategy.cast(s) for s in self._row_strategies]
        n = len(strategies)
        matrix = np.zeros((n, n), dtype=float)
        for i in range(n):
            for j in range(n):
                if i != j:
                    denom = strategies[i].x * strategies[j].y - strategies[j].x * strategies[i].y
                    if abs(denom) < 1e-15:
                        raise ValueError(
                            "x_i * y_j - x_j * y_i が 0 のため行列を構成できません: "
                            f"(i, j)=({i}, {j})"
                        )
                    matrix[i, j] = 1.0 / denom
        return matrix

    def get_x_values(self) -> np.ndarray:
        return np.array([CauchyLikeXYPureStrategy.cast(s).x for s in self._row_strategies])

    def get_y_values(self) -> np.ndarray:
        return np.array([CauchyLikeXYPureStrategy.cast(s).y for s in self._row_strategies])

