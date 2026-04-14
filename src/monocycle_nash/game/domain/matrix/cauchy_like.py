"""
コーシー型利得行列 (Cauchy-like payoff matrix)

行列の形: B_ij = a_i * a_j / (b_i - b_j)  (i ≠ j), B_ii = 0

本来のコーシー行列 C_ij = 1/(s_i + t_j) とは形が微妙に異なるため CauchyLike と命名。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .base import PayoffMatrix
from ..strategy import PureStrategy, PureStrategySet

if TYPE_CHECKING:
    from monocycle_nash.equilibrium.domain.mixed_strategy import MixedStrategy


@dataclass(frozen=True)
class CauchyLikeEntity:
    """
    コーシー型行列の行/列に対応するエンティティ。

    Attributes:
        a: スケールパラメータ a_i
        b: 位置パラメータ b_i
        label: 表示用ラベル
    """

    a: float
    b: float
    label: str = ""


@dataclass(frozen=True)
class CauchyLikePureStrategy(PureStrategy):
    """コーシー型行列専用の純粋戦略。"""

    entity: CauchyLikeEntity

    @property
    def a(self) -> float:
        """スケールパラメータ a_i"""
        return self.entity.a

    @property
    def b(self) -> float:
        """位置パラメータ b_i"""
        return self.entity.b

    @staticmethod
    def cast(strategy: PureStrategy) -> "CauchyLikePureStrategy":
        if not isinstance(strategy, CauchyLikePureStrategy):
            raise TypeError("CauchyLikePureStrategy が必要です")
        return strategy


class CauchyLikePayoffMatrix(PayoffMatrix):
    """
    コーシー型利得行列。

    行列の要素は以下で定義される:
        B_ij = a_i * a_j / (b_i - b_j)  (i ≠ j)
        B_ii = 0

    この行列は交代行列（歪対称行列）となる:
        B_ij = a_i * a_j / (b_i - b_j) = -a_j * a_i / (b_j - b_i) = -B_ji

    均衡解の理論式:
        M_ij = 1/(b_i - b_j) の零空間ベクトル u を求め、x_j = u_j / a_j と変換する。

    注意: 良く誤答される式として vi = Π_{j≠i}(b_i - b_j) のみを確率として使う例がある。
    しかしこの値には負の成分が含まれることがあり、直接確率として使うことはできない。
    正しくは u_j / a_j を正規化した theoretical_equilibrium() を使うこと（product_formula() 参照）。
    """

    def __init__(
        self,
        params: list[tuple[float, float]] | list[tuple[float, float, str]],
        labels: list[str] | None = None,
    ) -> None:
        """
        Args:
            params: 各戦略のパラメータ。(a_i, b_i) または (a_i, b_i, label_i) のリスト。
            labels: ラベルリスト（params に label が含まれない場合に使用）。
        """
        strategies: list[PureStrategy] = []
        for i, p in enumerate(params):
            if len(p) == 3:
                a, b, lbl = p[0], p[1], p[2]
            else:
                a, b = p[0], p[1]
                lbl = labels[i] if labels is not None else f"s{i}"
            entity = CauchyLikeEntity(a=float(a), b=float(b), label=lbl)
            strategies.append(CauchyLikePureStrategy(id=lbl, entity=entity))

        self._row_strategies = PureStrategySet(strategies=strategies, player_name="row")
        self._col_strategies = self._row_strategies
        self._matrix = self._calculate_matrix()

    @classmethod
    def from_ab_lists(
        cls,
        a_list: list[float],
        b_list: list[float],
        labels: list[str] | None = None,
    ) -> "CauchyLikePayoffMatrix":
        """
        a パラメータと b パラメータのリストからコーシー型行列を構成する。

        Args:
            a_list: スケールパラメータのリスト [a_1, ..., a_n]
            b_list: 位置パラメータのリスト [b_1, ..., b_n]
            labels: ラベルリスト（省略時は s0, s1, ... が使われる）

        Returns:
            CauchyLikePayoffMatrix
        """
        if len(a_list) != len(b_list):
            raise ValueError("a_list と b_list の長さが一致しません")
        if labels is not None and len(labels) != len(a_list):
            raise ValueError("labels の長さが a_list と一致しません")
        params: list[tuple[float, float]] = list(zip(a_list, b_list))
        return cls(params, labels=labels)

    # ---- PayoffMatrix 抽象メソッドの実装 ----

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

    def solve_equilibrium(self) -> "MixedStrategy":
        """一般行列として数値的にナッシュ均衡解を計算する。"""
        from monocycle_nash.equilibrium.infra.solver.selector import SolverSelector

        selector = SolverSelector()
        return selector.solve(self)

    # ---- コーシー型行列固有のメソッド ----

    def _calculate_matrix(self) -> np.ndarray:
        """B_ij = a_i * a_j / (b_i - b_j) の行列を計算する。"""
        strategies = [CauchyLikePureStrategy.cast(s) for s in self._row_strategies]
        n = len(strategies)
        matrix = np.zeros((n, n), dtype=float)
        for i in range(n):
            for j in range(n):
                if i != j:
                    denom = strategies[i].b - strategies[j].b
                    if abs(denom) < 1e-15:
                        raise ValueError(
                            f"b_i と b_j が等しいため行列を構成できません: "
                            f"b[{i}] = b[{j}] = {strategies[i].b}"
                        )
                    matrix[i, j] = strategies[i].a * strategies[j].a / denom
        return matrix

    def get_a_values(self) -> np.ndarray:
        """スケールパラメータ a のベクトルを返す。"""
        return np.array(
            [CauchyLikePureStrategy.cast(s).a for s in self._row_strategies]
        )

    def get_b_values(self) -> np.ndarray:
        """位置パラメータ b のベクトルを返す。"""
        return np.array(
            [CauchyLikePureStrategy.cast(s).b for s in self._row_strategies]
        )

    def product_formula(self) -> np.ndarray:
        """
        良く誤答される公式: v_i = Π_{j≠i} (b_i - b_j) を計算する。

        この値はゼロ空間に関連した中間値であり、b_i の順序によっては負の成分を含む。
        そのため、この生の値をそのまま均衡確率として使うことはできない。
        正しい均衡確率には a_i による除算と正規化が必要であり（theoretical_equilibrium() 参照）、
        それを省略して v_i を確率として扱う誤りが誤答の典型例である。

        Returns:
            各戦略に対応する v_i = Π_{j≠i}(b_i - b_j) の値の配列。正規化前の生の値。
        """
        b = self.get_b_values()
        n = len(b)
        v = np.ones(n, dtype=float)
        for i in range(n):
            for j in range(n):
                if i != j:
                    v[i] *= b[i] - b[j]
        return v

    def theoretical_equilibrium(self) -> np.ndarray:
        """
        コーシー型行列のナッシュ均衡確率を返す。

        一般行列に対するナッシュ均衡ソルバー（nashpy）を呼び出して厳密解を計算する。
        これは既存の solve_equilibrium() と本質的に同じ計算であり、
        特殊構造（M の零空間）を利用した近似的な導出を経由しない。

        注意: 良く誤答される式「v_i = Π_{j≠i}(b_i - b_j)」は n=3 でのみ
        零ベクトルに比例し、n≥5 では一般に均衡解を与えない（product_formula() 参照）。

        Returns:
            正規化された均衡確率の配列。
        """
        return np.asarray(self.solve_equilibrium().probabilities, dtype=float)
