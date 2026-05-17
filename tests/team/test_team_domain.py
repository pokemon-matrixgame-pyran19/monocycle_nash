from __future__ import annotations

import math

import pytest

from monocycle_nash.domain.character import Character, MatchupVector
from monocycle_nash.domain.strategy import PureStrategySet
from monocycle_nash.domain.team import Team


def test_team_calculate_feature_vector() -> None:
    strategies = PureStrategySet.from_characters(
        [
            Character(0.0, MatchupVector(2.0, 0.0), "A"),
            Character(0.0, MatchupVector(0.0, 1.0), "B"),
        ]
    )
    team = Team(label="A+B", member_ids=("c0", "c1"))

    feature = team.calculate_feature_vector(strategies)

    assert feature.x == pytest.approx(1.0)
    assert feature.y == pytest.approx(-0.5)
    assert feature.distance_from_origin == pytest.approx(math.hypot(1.0, -0.5))
    assert feature.angle_rad == pytest.approx(math.atan2(-0.5, 1.0))
    assert feature.angle_deg == pytest.approx(math.degrees(math.atan2(-0.5, 1.0)))


def test_team_calculate_feature_vector_requires_two_members() -> None:
    strategies = PureStrategySet.from_characters(
        [
            Character(0.0, MatchupVector(1.0, 0.0), "A"),
            Character(0.0, MatchupVector(0.0, 1.0), "B"),
            Character(0.0, MatchupVector(1.0, 1.0), "C"),
        ]
    )
    team = Team(label="A+B+C", member_ids=("c0", "c1", "c2"))

    with pytest.raises(ValueError, match="requires exactly 2 members"):
        team.calculate_feature_vector(strategies)


def test_team_calculate_feature_vector_rejects_zero_denominator() -> None:
    strategies = PureStrategySet.from_characters(
        [
            Character(0.0, MatchupVector(1.0, 0.0), "A"),
            Character(0.0, MatchupVector(2.0, 0.0), "B"),
        ]
    )
    team = Team(label="A+B", member_ids=("c0", "c1"))

    with pytest.raises(ValueError, match="denominator is zero"):
        team.calculate_feature_vector(strategies)
