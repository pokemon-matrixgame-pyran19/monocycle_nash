"""
キャラクター関連モジュール

単相性モデル用のキャラクターと相性ベクトルを提供します。
"""

from .domain import MatchupVector, Character#, get_characters
from .calculator import PowerVectorCalculator

__all__ = [
    "MatchupVector",
    "Character",
    #"get_characters",
    "PowerVectorCalculator",
]