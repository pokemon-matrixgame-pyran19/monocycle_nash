import pytest
from rule.gain_matrix import Environment
from rule.character import MatchupVector, Character
import random
import numpy as np
import numpy.testing as npt

ROOT3=1.732050


def test_GainMatrix_set_gain():
    """
    等パワーで正三角形のベクトルに対して、利得行列を計算
    """
    power = random.random()
    c1=[power  ,  2,      0]
    c2=[power  , -1,  ROOT3]
    c3=[power+1, -1, -ROOT3]

    A = [[0,1,-1], [-1,0,1], [1,-1,0]]
    P3 = [[0,0,-1],[0,0,-1], [1,1,0]] # c3がパワー高い分の補正

    # c1,c2,c3に対してA*2ROOT3の行列が返ることを確認

    characters = [
        Character(c1[0], MatchupVector(c1[1], c1[2])),
        Character(c2[0], MatchupVector(c2[1], c2[2])),
        Character(c3[0], MatchupVector(c3[1], c3[2])),
    ]

    env = Environment(characters)
    gainMatrix = env.get_matrix()

    npA=2*ROOT3*np.array(A) + np.array(P3)

    assert gainMatrix.shape == (3,3)
    # 行列の一致を確認。　書き方色々あるみたい？
    assert gainMatrix == pytest.approx(npA, rel=1e-4)
    # npt.assert_allclose(gainMatrix, npA, rtol=1e-4, atol=1e-4) 

def test_env_equal():
    """
    等価判定のテスト
    等価なはずのenv同士で==が等価判定として機能するか確認
    """

    power1 = random.random()
    c1=[power1,  2,      0]
    c2=[power1, -1,  ROOT3]
    c3=[power1, -1, -ROOT3]

    power2 = random.random()
    d1=[power2,      0,  2]
    d2=[power2, -ROOT3, -1]
    d3=[power2,  ROOT3, -1]

    characters1 = [
        Character(c1[0], MatchupVector(c1[1], c1[2])),
        Character(c2[0], MatchupVector(c2[1], c2[2])),
        Character(c3[0], MatchupVector(c3[1], c3[2])),
    ]

    characters2 = [
        Character(d1[0], MatchupVector(d1[1], d1[2])),
        Character(d2[0], MatchupVector(d2[1], d2[2])),
        Character(d3[0], MatchupVector(d3[1], d3[2])),
    ]

    characters3 = [
        Character(0, MatchupVector(0, 0)),
        Character(0, MatchupVector(0, 0)),
        Character(0, MatchupVector(0, 0)),
    ]

    env1 = Environment(characters1)
    env2 = Environment(characters2)
    env3 = Environment(characters3)
    assert env1 == env1
    assert env1 == env2
    assert env1 != env3

def test_env_convert():

    c1=[0.25,    2,      0]
    c2=[0.10,   -1,      1]
    c3=[0.00,   -1,     -1]

    # a=   (1,0)
    # p' = p + v x a
    # v' = v- a

    d1=[0.25+0,   1.0,       0]   
    d2=[0.1-1 ,  -2.0,       1]
    d3=[0+1   ,  -2.0,      -1]

    characters1 = [
        Character(c1[0], MatchupVector(c1[1], c1[2])),
        Character(c2[0], MatchupVector(c2[1], c2[2])),
        Character(c3[0], MatchupVector(c3[1], c3[2])),
    ]

    characters2 = [
        Character(d1[0], MatchupVector(d1[1], d1[2])),
        Character(d2[0], MatchupVector(d2[1], d2[2])),
        Character(d3[0], MatchupVector(d3[1], d3[2])),
    ]

    env1 = Environment(characters1)
    env2 = Environment(characters2)

    assert env1 == env2
    assert env1.convert([1,0]) == env1

    
def test_isopower():

    c1=[0.25,    2,      0]
    c2=[0.10,   -1,      1]
    c3=[0.00,   -1,     -1]

    # a=   0.25*(0,2)+0.10*(-3,-1) =(-0.3,0.4)/D 
    # D=det (c1,c2,c3)=0.25*(1-(-1))+0.10*(-2*(-1))+0*(xxx)=0.7

    # dの値手計算出来てないので後でちゃんと設定する
    d1=[0,   2.3,    -0.4]   
    d2=[0,  -0.7,  0.6]
    d3=[0, -0.35, -1.4]

    characters1 = [
        Character(c1[0], MatchupVector(c1[1], c1[2])),
        Character(c2[0], MatchupVector(c2[1], c2[2])),
        Character(c3[0], MatchupVector(c3[1], c3[2])),
    ]

    characters2 = [
        Character(d1[0], MatchupVector(d1[1], d1[2])),
        Character(d2[0], MatchupVector(d2[1], d2[2])),
        Character(d3[0], MatchupVector(d3[1], d3[2])),
    ]

    env1 = Environment(characters1)
    env2 = Environment(characters2)

    assert env1 == env2