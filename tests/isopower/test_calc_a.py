from isopower.calc_a import aCalculator, DevCalculator
from rule.character import Character, MatchupVector
import pytest
import numpy as np

@pytest.mark.dev()
def test_vertical():
    c1 = Character(10, MatchupVector( 0,  0))
    c2 = Character( 0, MatchupVector(10,  0))
    c3 = Character( 0, MatchupVector( 0, 10))

    calc = DevCalculator(c1, c2, c3)
    a = calc.dev_calc()
    assert calc.v31 == np.array(10,  0, -10)
    assert calc.v21 == np.array( 0, 10, -10)
    assert calc.nu  == np.array(10, 10,  100)

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
    a = -calc.calc()

    assert a.x == pytest.approx( expected_a[0])
    assert a.y == pytest.approx(expected_a[1])