from isopower.optimal_triangle import OptimalTriangleFinder
import numpy as np
from rule.character import get_characters

def countup():
    # 正六面体
    points = np.array([
        [  0, 0, 0],
        [  1, 0, 0],
        [  0, 1, 0],
        [  0, 0, 1],
        [ -1, 0, 0],
        [  0,-1, 0],
        [  0, 0,-1]
    ])
    points = np.array([
        [ 10,  2,         0],
        [ 10, -1,  1.732050],
        [ 10, -1, -1.732050],
        [  0,  2,         2],
        [ -1, -1,       3.3],
        [  0, -1,        -1],
        [  0,  1,      -0.4]
    ])
    characters=get_characters(points)
    finder=OptimalTriangleFinder(characters)
    finder.find()
    finder.display()



if __name__ == "__main__":
    countup()