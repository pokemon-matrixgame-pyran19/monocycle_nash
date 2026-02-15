from __future__ import annotations

from abc import ABC, abstractmethod

import nashpy as nash
import numpy as np

from ..matrix.base import PayoffMatrix
from ..matrix.general import GeneralPayoffMatrix
from ..matrix.monocycle import MonocyclePayoffMatrix
from ..strategy.domain import MonocyclePureStrategy, PureStrategySet
from .domain import Team


class TwoByTwoGameValueCalculator:
    @staticmethod
    def calculate(matrix: np.ndarray) -> float:
        if matrix.shape != (2, 2):
            raise ValueError("2x2 matrix is required")

        saddle = TwoByTwoGameValueCalculator.calculate_saddle_point(matrix)
        if saddle is not None:
            return saddle[0]

        a, b = matrix[0, 0], matrix[0, 1]
        c, d = matrix[1, 0], matrix[1, 1]
        denominator = a + d - b - c
        if abs(denominator) < 1e-12:
            return float(np.mean(matrix))
        return float((a * d - b * c) / denominator)

    @staticmethod
    def calculate_saddle_point(matrix: np.ndarray) -> tuple[float, float] | None:
        row_mins = np.min(matrix, axis=1)
        col_maxs = np.max(matrix, axis=0)
        maximin = np.max(row_mins)
        minimax = np.min(col_maxs)
        if abs(maximin - minimax) > 1e-10:
            return None
        return float(maximin), float(minimax)


class MonocycleTwoByTwoValueCalculator:
    """MonocyclePureStrategy を入力に取る単相性2x2理論値計算。"""

    def __init__(
        self,
        s1: MonocyclePureStrategy,
        s2: MonocyclePureStrategy,
        s3: MonocyclePureStrategy,
        s4: MonocyclePureStrategy,
    ):
        self.s1, self.s2, self.s3, self.s4 = s1, s2, s3, s4

    def calculate_e_parameter(self) -> float:
        return self.s1.power - self.s2.power + self.s1.vector.times(self.s2.vector)

    def calculate_f_parameter(self) -> float:
        return self.s3.power - self.s4.power + self.s3.vector.times(self.s4.vector)

    def calculate_m_determinant(self) -> float:
        mat = np.array(
            [
                [1.0, 1.0, 1.0, 1.0],
                [self.s1.power, self.s2.power, self.s3.power, self.s4.power],
                [self.s1.vector.x, self.s2.vector.x, self.s3.vector.x, self.s4.vector.x],
                [self.s1.vector.y, self.s2.vector.y, self.s3.vector.y, self.s4.vector.y],
            ],
            dtype=float,
        )
        return float(np.linalg.det(mat))

    def calculate_game_value(self) -> float:
        e = self.calculate_e_parameter()
        f = self.calculate_f_parameter()
        m = self.calculate_m_determinant()
        denom = (self.s1.vector - self.s2.vector).times(self.s3.vector - self.s4.vector)
        if abs(denom) < 1e-12:
            return float((e + f) / 2.0)
        return float((e * f + m) / denom)


class TeamPayoffCalculator(ABC):
    @abstractmethod
    def calculate(self, team1: Team, team2: Team, char_matrix: PayoffMatrix) -> float:
        raise NotImplementedError


class ExactTeamPayoffCalculator(TeamPayoffCalculator):
    def calculate(self, team1: Team, team2: Team, char_matrix: PayoffMatrix) -> float:
        rows = team1.resolve_member_indices(char_matrix.row_strategies)
        cols = team2.resolve_member_indices(char_matrix.col_strategies)
        sub = char_matrix.matrix[np.ix_(rows, cols)]

        game = nash.Game(sub)
        sigma_row, sigma_col = game.linear_program()
        return float(np.asarray(sigma_row) @ sub @ np.asarray(sigma_col))


class TwoByTwoFormulaCalculator(TeamPayoffCalculator):
    def calculate(self, team1: Team, team2: Team, char_matrix: PayoffMatrix) -> float:
        rows = team1.resolve_member_indices(char_matrix.row_strategies)
        cols = team2.resolve_member_indices(char_matrix.col_strategies)
        if len(rows) != 2 or len(cols) != 2:
            raise ValueError("TwoByTwoFormulaCalculator requires 2x2 team matchup")
        sub = char_matrix.matrix[np.ix_(rows, cols)]
        return TwoByTwoGameValueCalculator.calculate(sub)


class MonocycleFormulaCalculator(TeamPayoffCalculator):
    def calculate(self, team1: Team, team2: Team, char_matrix: PayoffMatrix) -> float:
        if not isinstance(char_matrix, MonocyclePayoffMatrix):
            raise TypeError("MonocyclePayoffMatrix is required")
        rows = team1.resolve_member_indices(char_matrix.row_strategies)
        cols = team2.resolve_member_indices(char_matrix.col_strategies)
        if len(rows) != 2 or len(cols) != 2:
            raise ValueError("MonocycleFormulaCalculator requires 2x2 team matchup")

        sub = char_matrix.matrix[np.ix_(rows, cols)]

        # 理論式は4キャラが互いに異なり、かつ純粋戦略均衡を持たないケースで導出される。
        # 条件外では一般2x2公式（サドル判定込み）へフォールバックする。
        if set(team1.member_ids) & set(team2.member_ids):
            return TwoByTwoGameValueCalculator.calculate(sub)
        if TwoByTwoGameValueCalculator.calculate_saddle_point(sub) is not None:
            return TwoByTwoGameValueCalculator.calculate(sub)

        s1 = MonocyclePureStrategy.cast(char_matrix.row_strategies[rows[0]])
        s2 = MonocyclePureStrategy.cast(char_matrix.row_strategies[rows[1]])
        s3 = MonocyclePureStrategy.cast(char_matrix.col_strategies[cols[0]])
        s4 = MonocyclePureStrategy.cast(char_matrix.col_strategies[cols[1]])
        return MonocycleTwoByTwoValueCalculator(s1, s2, s3, s4).calculate_game_value()


class TeamPayoffCalculatorSelector:
    def __init__(self, use_monocycle_formula: bool = True):
        self.use_monocycle_formula = use_monocycle_formula

    def select_calculator(self, team1: Team, team2: Team, char_matrix: PayoffMatrix) -> TeamPayoffCalculator:
        row_n = len(team1.member_ids)
        col_n = len(team2.member_ids)
        if row_n == 2 and col_n == 2:
            if self.use_monocycle_formula and isinstance(char_matrix, MonocyclePayoffMatrix):
                return MonocycleFormulaCalculator()
            return TwoByTwoFormulaCalculator()
        return ExactTeamPayoffCalculator()

    def calculate(self, team1: Team, team2: Team, char_matrix: PayoffMatrix) -> float:
        calculator = self.select_calculator(team1, team2, char_matrix)
        return calculator.calculate(team1, team2, char_matrix)


class TwoPlayerTeamMatrixCalculator:
    def __init__(self, character_matrix: PayoffMatrix, use_monocycle_formula: bool = True):
        self.character_matrix = character_matrix
        self.selector = TeamPayoffCalculatorSelector(use_monocycle_formula=use_monocycle_formula)

    def calculate_team_value(self, team1: Team, team2: Team) -> float:
        return self.selector.calculate(team1, team2, self.character_matrix)

    def generate_matrix(self, teams: list[Team]) -> GeneralPayoffMatrix:
        n = len(teams)
        matrix = np.zeros((n, n), dtype=float)
        for i in range(n):
            for j in range(i + 1, n):
                value = self.calculate_team_value(teams[i], teams[j])
                matrix[i, j] = value
                matrix[j, i] = -value
        row_strategies = PureStrategySet.from_teams(teams, player_name="row")
        return GeneralPayoffMatrix(matrix, row_strategies, row_strategies)
