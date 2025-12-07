from isopower.calc_a import aCalculator, DevCalculator
from rule.character import Character, MatchupVector
import pytest
import numpy as np

def test_calc_a():
    """
    以下二つは等価  a=(0.5,0.5)
    [[0, 2, 0], [0, -1, 1.73205], [0, -1, -1.73205]]
    [[1.0, 1.5, -0.5], [-1.366025, -1.5, 1.23205], [0.36602500000000004, -1.5, -2.23205]]
    """

    c1 = Character(1, MatchupVector(1.5, -0.5))
    c2 = Character(-1.366025, MatchupVector(-1.5, 1.23205))
    c3 = Character(0.366025, MatchupVector(-1.5, -2.23205))

    expected_a = [-0.5, -0.5]

    calc = aCalculator(c1, c2, c3)
    a = calc.calc()

    assert a.x == pytest.approx(expected_a[0])
    assert a.y == pytest.approx(expected_a[1])

def test_inner():
    c1 = Character(1, MatchupVector(2, 0))
    c2 = Character(0, MatchupVector(-1, 1.732050))
    c3 = Character(0, MatchupVector(-1, -1.732050))

    calc = aCalculator(c1, c2, c3)
    a = calc.calc()

    assert calc.is_inner == True
    assert calc.is_edge == False

def test_ounter():
    ROOT3 = 1.7320508075688772
    c1 = Character(25*2*ROOT3, MatchupVector(2, 0))
    c2 = Character(10*2*ROOT3, MatchupVector(-1, 1.732050))
    c3 = Character(         0, MatchupVector(-1, -1.732050))

    calc = aCalculator(c1, c2, c3)
    a = calc.calc()

    assert calc.is_inner == False
    assert calc.is_edge == False

def test_edge():
    ROOT3 = 1.7320508075688772
    c1 = Character(       0, MatchupVector(  2,     0))
    c2 = Character( 2*ROOT3, MatchupVector( -1, ROOT3))
    c3 = Character(-2*ROOT3, MatchupVector( -1,-ROOT3))
    calc = aCalculator(c1, c2, c3)
    a = calc.calc()

    assert calc.is_edge == True
    assert calc.is_inner == True # 今のところ境界上の時は内側とみなす仕様