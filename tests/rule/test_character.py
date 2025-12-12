import pytest
from rule.gain_matrix import Pool
from rule.character import MatchupVector, Character
import random
import numpy as np
import numpy.testing as npt

ROOT3=1.732050


def test_Matchup_times():
    """
    外積計算のテスト
    正三角形の3点に対して反時計回りに2√3になることを確認
    """
    v1 = [2 , 0]
    v2 = [-1 , ROOT3]
    v3 = [-1 ,-ROOT3]
    v4 = [0 , 3]

    mv1 = MatchupVector(v1[0], v1[1])
    mv2 = MatchupVector(v2[0], v2[1])
    mv3 = MatchupVector(v3[0], v3[1])
    mv4 = MatchupVector(v4[0], v4[1])

    assert mv1.times(mv2) == pytest.approx(2*ROOT3, rel=1e-4)
    assert mv2.times(mv3) == pytest.approx(2*ROOT3, rel=1e-4)
    assert mv3.times(mv1) == pytest.approx(2*ROOT3, rel=1e-4)
    assert mv1.times(mv4) == pytest.approx(6, rel=1e-4)

def test_Matchup_reverse():
    """
    外積の交換法則
    入れ替えて負になることを確認
    """
    v1 = MatchupVector(random.random(), random.random())
    v2 = MatchupVector(random.random(), random.random())

    assert v1.times(v1) == 0
    assert v1.times(v2) == - v2.times(v1)
