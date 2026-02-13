from __future__ import annotations

from abc import ABC, abstractmethod

import nashpy as nash
import numpy as np

from ..character.domain import Character
from ..matrix.base import PayoffMatrix
from ..matrix.general import GeneralPayoffMatrix
from ..matrix.monocycle import MonocyclePayoffMatrix
from ..strategy.domain import PureStrategySet
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
    def __init__(self, c1: Character, c2: Character, c3: Character, c4: Character):
        self.c1, self.c2, self.c3, self.c4 = c1, c2, c3, c4

    def calculate_e_parameter(self) -> float:
        return self.c1.p - self.c2.p + self.c1.v.times(self.c2.v)

    def calculate_f_parameter(self) -> float:
        return self.c3.p - self.c4.p + self.c3.v.times(self.c4.v)

    def calculate_m_determinant(self) -> float:
        mat = np.array(
            [
                [1.0, 1.0, 1.0, 1.0],
                [self.c1.p, self.c2.p, self.c3.p, self.c4.p],
                [self.c1.v.x, self.c2.v.x, self.c3.v.x, self.c4.v.x],
                [self.c1.v.y, self.c2.v.y, self.c3.v.y, self.c4.v.y],
            ],
            dtype=float,
        )
        return float(np.linalg.det(mat))

    def calculate_game_value(self) -> float:
        e = self.calculate_e_parameter()
        f = self.calculate_f_parameter()
        m = self.calculate_m_determinant()
        denom = (self.c1.v - self.c2.v).times(self.c3.v - self.c4.v)
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
        equilibria = list(game.linear_program())
        if not equilibria:
            return float(np.mean(sub))
        sigma_row, sigma_col = equilibria[0]
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

        c1 = char_matrix.row_strategies[rows[0]].entity
        c2 = char_matrix.row_strategies[rows[1]].entity
        c3 = char_matrix.col_strategies[cols[0]].entity
        c4 = char_matrix.col_strategies[cols[1]].entity
        return MonocycleTwoByTwoValueCalculator(c1, c2, c3, c4).calculate_game_value()


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
