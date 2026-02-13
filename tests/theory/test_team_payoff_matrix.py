"""理論予測テスト - 構築(Team)利得行列の検証。"""

import numpy as np
import pytest

from monocycle_nash.character.domain import Character, MatchupVector
from monocycle_nash.matrix.builder import PayoffMatrixBuilder
from monocycle_nash.team.domain import Team

from theory import TeamTheoryTestCase, TheoryTestBuilder


class TestTeamPayoffMatrixTheory:
    def _build_characters(self, case: TeamTheoryTestCase) -> list[Character]:
        variant = case.character_case.primary_variant
        return [
            Character(power=p, vector=MatchupVector(vx, vy))
            for p, (vx, vy) in zip(variant.powers, variant.vectors)
        ]

    def _build_teams(self, case: TeamTheoryTestCase) -> list[Team]:
        return [
            Team(label=label, member_ids=members)
            for label, members in zip(case.team_labels, case.team_members)
        ]

    def test_all_team_cases_matrix_matches_theory(self):
        for case in TheoryTestBuilder.get_all_team_cases():
            characters = self._build_characters(case)
            teams = self._build_teams(case)
            character_matrix = PayoffMatrixBuilder.from_characters(characters)

            team_matrix = PayoffMatrixBuilder.from_team_matchups(
                teams=teams,
                character_matrix=character_matrix,
                use_monocycle_approx=False,
            )

            np.testing.assert_array_almost_equal(
                team_matrix.matrix,
                case.matrix,
                decimal=10,
                err_msg=f"{case.name}: 構築利得行列が理論値と一致しない",
            )

    def test_all_team_cases_matrix_is_skew_symmetric(self):
        for case in TheoryTestBuilder.get_all_team_cases():
            characters = self._build_characters(case)
            teams = self._build_teams(case)
            character_matrix = PayoffMatrixBuilder.from_characters(characters)
            team_matrix = PayoffMatrixBuilder.from_team_matchups(
                teams=teams,
                character_matrix=character_matrix,
                use_monocycle_approx=False,
            )

            matrix = team_matrix.matrix
            n = matrix.shape[0]
            for i in range(n):
                assert matrix[i, i] == pytest.approx(0.0, abs=1e-10)
                for j in range(n):
                    assert matrix[i, j] == pytest.approx(-matrix[j, i], abs=1e-10)

    def test_all_team_cases_labels_are_fixed_order(self):
        for case in TheoryTestBuilder.get_all_team_cases():
            characters = self._build_characters(case)
            teams = self._build_teams(case)
            character_matrix = PayoffMatrixBuilder.from_characters(characters)
            team_matrix = PayoffMatrixBuilder.from_team_matchups(
                teams=teams,
                character_matrix=character_matrix,
                use_monocycle_approx=False,
            )

            assert team_matrix.labels == case.team_labels
