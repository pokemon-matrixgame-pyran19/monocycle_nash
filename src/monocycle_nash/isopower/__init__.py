"""
等パワー座標関連モジュール

単相性モデルでの等パワー座標計算と最適三角形探索を提供します。
"""

from .calc_a import aCalculator
from .triangle import OptimalTriangleFinder, OptimalTriangleResult

__all__ = ["aCalculator", "OptimalTriangleFinder", "OptimalTriangleResult"]
