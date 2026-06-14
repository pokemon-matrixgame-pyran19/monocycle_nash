from __future__ import annotations
import math
import pytest
from monocycle_nash.application.matrix_nodes import CharacterRotatingPairSource
from monocycle_nash.domain.character import MatchupVector

def test_rotating_pair_coordinates_should_be_cos_sin() -> None:
    # theta=0, d=pi/2
    # v3 = (r3*cos(0), r3*sin(0)) = (1.0, 0.0)
    # v4 = (r4*cos(pi/2), r4*sin(pi/2)) = (0.0, 2.0)
    source = CharacterRotatingPairSource(
        x=10.0,
        y=20.0,
        r3=1.0,
        r4=2.0,
        d=math.pi / 2,
        theta_step_rad=math.pi,  # Two steps to keep it simple
        theta_offset_rad=0.0,
        power=0.0
    )

    # CharacterRotatingPairSource.load_characters doesn't actually use the ctx argument
    characters = source.load_characters(None)  # type: ignore

    # characters[0] -> fixed_label_1 (c1)
    # characters[1] -> fixed_label_2 (c2)
    # characters[2] -> g3_000
    # characters[3] -> g4_000

    v3_0 = characters[2].v
    v4_0 = characters[3].v

    # These assertions will FAIL with the current implementation
    assert v3_0.x == pytest.approx(1.0), f"v3_0.x expected 1.0, got {v3_0.x}"
    assert v3_0.y == pytest.approx(0.0), f"v3_0.y expected 0.0, got {v3_0.y}"
    assert v4_0.x == pytest.approx(0.0), f"v4_0.x expected 0.0, got {v4_0.x}"
    assert v4_0.y == pytest.approx(2.0), f"v4_0.y expected 2.0, got {v4_0.y}"

def test_rotating_pair_rotation_direction() -> None:
    # theta=pi/2, d=0
    # v3 = (r3*cos(pi/2), r3*sin(pi/2)) = (0.0, 1.0)
    source = CharacterRotatingPairSource(
        x=10.0,
        y=20.0,
        r3=1.0,
        r4=1.0,
        d=0.0,
        theta_step_rad=math.pi,
        theta_offset_rad=math.pi / 2,
        power=0.0
    )

    characters = source.load_characters(None)  # type: ignore
    v3_0 = characters[2].v

    # These assertions will FAIL with the current implementation
    assert v3_0.x == pytest.approx(0.0)
    assert v3_0.y == pytest.approx(1.0)
