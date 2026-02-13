"""
パワー・ベクトル計算モジュール

キャラクターのパワーと相性ベクトルに関する計算機能を提供します。
"""

from __future__ import annotations
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .domain import Character, MatchupVector


class PowerVectorCalculator:
    """
    パワー・ベクトル計算クラス
    
    キャラクターグループの統計情報や変換を提供します。
    """
    
    @staticmethod
    def get_power_vector(characters: list[Character]) -> np.ndarray:
        """
        キャラクターリストからパワーベクトルを取得
        
        Args:
            characters: Characterオブジェクトのリスト
        
        Returns:
            パワー値のnumpy配列 [p1, p2, ..., pn]
        """
        return np.array([c.p for c in characters])
    
    @staticmethod
    def get_matchup_vectors(characters: list[Character]) -> np.ndarray:
        """
        キャラクターリストから相性ベクトル行列を取得
        
        Args:
            characters: Characterオブジェクトのリスト
        
        Returns:
            相性ベクトルのnumpy配列 (N×2) [[x1, y1], [x2, y2], ...]
        """
        if not characters:
            return np.empty((0, 2), dtype=float)
        return np.array([[c.v.x, c.v.y] for c in characters], dtype=float)
    
    @staticmethod
    def calculate_power_stats(characters: list[Character]) -> dict[str, float]:
        """
        パワー値の統計情報を計算
        
        Args:
            characters: Characterオブジェクトのリスト
        
        Returns:
            {
                "mean": 平均値,
                "std": 標準偏差,
                "min": 最小値,
                "max": 最大値
            }
        """
        powers = PowerVectorCalculator.get_power_vector(characters)
        if powers.size == 0:
            raise ZeroDivisionError("空リストの統計は計算できません")
        return {
            "mean": float(np.mean(powers)),
            "std": float(np.std(powers)),
            "min": float(np.min(powers)),
            "max": float(np.max(powers))
        }
    
    @staticmethod
    def find_min_power_index(characters: list[Character]) -> int:
        """
        最小パワーを持つキャラクターのインデックスを取得
        
        Args:
            characters: Characterオブジェクトのリスト
        
        Returns:
            最小パワーのキャラクターのインデックス
        """
        powers = PowerVectorCalculator.get_power_vector(characters)
        return int(np.argmin(powers))
    
    @staticmethod
    def find_max_power_index(characters: list[Character]) -> int:
        """
        最大パワーを持つキャラクターのインデックスを取得
        
        Args:
            characters: Characterオブジェクトのリスト
        
        Returns:
            最大パワーのキャラクターのインデックス
        """
        powers = PowerVectorCalculator.get_power_vector(characters)
        return int(np.argmax(powers))
    
    @staticmethod
    def calculate_centroid(characters: list[Character]) -> MatchupVector:
        """
        相性ベクトルの重心を計算
        
        Args:
            characters: Characterオブジェクトのリスト
        
        Returns:
            重心のMatchupVector
        """
        from .domain import MatchupVector
        
        if not characters:
            raise ValueError("キャラクターリストが空です")
        
        vectors = PowerVectorCalculator.get_matchup_vectors(characters)
        centroid = np.mean(vectors, axis=0)
        return MatchupVector(centroid[0], centroid[1])
    
    @staticmethod
    def shift_to_origin(characters: list[Character], 
                        origin: MatchupVector | list[float] | None = None) -> list[Character]:
        """
        全キャラクターの相性ベクトルを原点移動
        
        Args:
            characters: Characterオブジェクトのリスト
            origin: 新しい原点（Noneの場合は重心を使用）
        
        Returns:
            原点移動後のCharacterリスト
        """
        from .domain import MatchupVector
        
        if not characters:
            return []
        
        if origin is None:
            origin = PowerVectorCalculator.calculate_centroid(characters)
        else:
            origin = MatchupVector(origin)
        
        return [c.convert(origin) for c in characters]
