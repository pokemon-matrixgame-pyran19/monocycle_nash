import numpy as np
from typing import TYPE_CHECKING

from .general import GeneralPayoffMatrix
from .monocycle import MonocyclePayoffMatrix
from rule.character import Character

if TYPE_CHECKING:
    pass  # Team関連は後で実装


class PayoffMatrixBuilder:
    """
    利得行列ビルダー
    - CharacterからMonocyclePayoffMatrixを生成
    """
    
    @staticmethod
    def from_characters(
        characters: list[Character], 
        labels: list[str] | None = None
    ) -> MonocyclePayoffMatrix:
        """Characterリストから単相性モデル利得行列を生成"""
        return MonocyclePayoffMatrix(characters, labels)
    
    @staticmethod
    def from_general_matrix(
        matrix: np.ndarray, 
        labels: list[str] | None = None
    ) -> GeneralPayoffMatrix:
        """任意の行列から一般利得行列を生成"""
        return GeneralPayoffMatrix(matrix, labels)
