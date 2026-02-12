"""
ゲームの値の計算にかかわるもの
"""
from __future__ import annotations
import numpy as np


def calc_score(A:np.ndarray,v1:np.ndarray,v2:np.ndarray):
    return v1.dot(A.dot(v2))


def calc_weight(A:np.ndarray,v:np.ndarray):
    return A.dot(v)