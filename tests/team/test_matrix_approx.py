import numpy as np

from monocycle_nash.character.domain import Character, MatchupVector
from monocycle_nash.matrix.builder import PayoffMatrixBuilder
from monocycle_nash.team.domain import Team
from monocycle_nash.team.matrix_approx import (
    TwoByTwoGameValueCalculator,
    TwoPlayerTeamMatrixCalculator,
)


def test_two_by_two_formula_matches_known_value():
    matrix = np.array([[1.0, -2.0], [3.0, 0.0]])
    value = TwoByTwoGameValueCalculator.calculate(matrix)
    expected = (1.0 * 0.0 - (-2.0) * 3.0) / (1.0 + 0.0 - (-2.0) - 3.0)
    assert value == expected


def test_generate_team_matrix_from_character_matrix():
    characters = [
        Character(1.0, MatchupVector(1.0, 0.0), label="A"),
        Character(0.2, MatchupVector(0.0, 1.0), label="B"),
        Character(0.8, MatchupVector(-1.0, 0.0), label="C"),
        Character(0.3, MatchupVector(0.0, -1.0), label="D"),
    ]
    char_matrix = PayoffMatrixBuilder.from_characters(characters)

    teams = [
        Team(label="T1", member_ids=("c0", "c1")),
        Team(label="T2", member_ids=("c2", "c3")),
    ]

    matrix_calculator = TwoPlayerTeamMatrixCalculator(char_matrix, use_monocycle_formula=False)
    matrix = matrix_calculator.generate_matrix(teams)

    sub = char_matrix.matrix[np.ix_([0, 1], [2, 3])]
    expected = TwoByTwoGameValueCalculator.calculate(sub)

    assert matrix.matrix.shape == (2, 2)
    assert matrix.matrix[0, 1] == expected
    assert matrix.matrix[1, 0] == -expected
    assert matrix.labels == ["T1", "T2"]


def test_builder_from_team_matchups_shortcut():
    characters = [
        Character(0.7, MatchupVector(1.0, 1.0)),
        Character(0.1, MatchupVector(-1.0, 1.0)),
        Character(0.4, MatchupVector(-1.0, -1.0)),
        Character(0.9, MatchupVector(1.0, -1.0)),
    ]
    char_matrix = PayoffMatrixBuilder.from_characters(characters)
    teams = [
        Team(label="Left", member_ids=(0, 1)),
        Team(label="Right", member_ids=(2, 3)),
    ]

    team_matrix = PayoffMatrixBuilder.from_team_matchups(
        teams=teams,
        character_matrix=char_matrix,
        use_monocycle_formula=False,
    )

    assert team_matrix.size == 2
    assert team_matrix.labels == ["Left", "Right"]
