from isopower.optimal_triangle import OptimalTriangleFinder
from rule.gain_matrix import Pool
from rule.character import get_characters
import numpy as np

ROOT3 = 1.7320508075688772

def test_finder():
    characters = get_characters(np.array([
        [      0,  1,        0],
        [ -ROOT3, -2,    ROOT3],
        [  ROOT3, -2,   -ROOT3],
        [ -ROOT3, -3, -2*ROOT3],
        [2*ROOT3,  3, -3*ROOT3],
        [   -0.6,  0,        0]
    ]))

    # expected = get_characters(np.array([
    #     [      0,  2,        0],
    #     [ -ROOT3, -1,    ROOT3],
    #     [  ROOT3, -1,   -ROOT3],
    #     [ -ROOT3, -2, -2*ROOT3],
    #     [2*ROOT3,  4, -3*ROOT3],
    #     [   -0.6,  1,        0]
    # ]))

    pool = Pool(characters)
    finder = OptimalTriangleFinder(pool)
    finder.find()
    result=finder.get_result()
    assert len(result)==1
    assert result[0][0]==[-1,0]
    assert result[0][1] in characters[:3]
    assert result[0][2] in characters[:3]
    assert result[0][3] in characters[:3]




