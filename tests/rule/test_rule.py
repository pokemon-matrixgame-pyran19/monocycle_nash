import pytest
from rule.gain_matrix import GainMatrix
from rule.character import MatchupVector, Character
import random
import numpy as np
import numpy.testing as npt

ROOT3=1.732050


def test_Matchup_times():
    """
    外積計算のテスト
    正三角形の3点に対して反時計回りに√3になることを確認
    """
    v1 = [2 , 0]
    v2 = [-1 , ROOT3]
    v3 = [-1 ,-ROOT3]

    mv1 = MatchupVector(v1[0], v1[1])
    mv2 = MatchupVector(v2[0], v2[1])
    mv3 = MatchupVector(v3[0], v3[1])

    assert mv1.times(mv2) == pytest.approx(2*ROOT3, rel=1e-4)
    assert mv2.times(mv3) == pytest.approx(2*ROOT3, rel=1e-4)
    assert mv3.times(mv1) == pytest.approx(2*ROOT3, rel=1e-4)

def test_Matchup_reverse():
    """
    外積の交換法則
    入れ替えて負になることを確認
    """
    v1 = MatchupVector(random.random(), random.random())
    v2 = MatchupVector(random.random(), random.random())

    assert v1.times(v1) == 0
    assert v1.times(v2) == - v2.times(v1)

def test_GainMatrix_set_gain():
    """
    等パワーで正三角形のベクトルに対して、利得行列を計算
    """
    power = random.random()
    c1=[power,  2,      0]
    c2=[power, -1,  ROOT3]
    c3=[power, -1, -ROOT3]

    A = [[0,1,-1], [-1,0,1], [1,-1,0]]

    # c1,c2,c3に対してA*ROOT3の行列が返ることを確認

    characters = [
        Character(c1[0], MatchupVector(c1[1], c1[2])),
        Character(c2[0], MatchupVector(c2[1], c2[2])),
        Character(c3[0], MatchupVector(c3[1], c3[2])),
    ]

    gm = GainMatrix(characters)
    gainMatrix = gm.get_matrix()

    npA=2*ROOT3*np.array(A)

    assert gainMatrix.shape == (3,3)
    # 行列の一致を確認。　書き方色々あるみたい？
    assert gainMatrix == pytest.approx(npA, rel=1e-4)
    # npt.assert_allclose(gainMatrix, npA, rtol=1e-4, atol=1e-4) 
