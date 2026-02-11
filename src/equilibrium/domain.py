import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from matrix.base import PayoffMatrix

class MixedStrategy:
    """混合戦略（ナッシュ均衡解）"""

    def __init__(self, probabilities: np.ndarray, strategy_ids: list[str]):
        if len(probabilities) != len(strategy_ids):
            raise ValueError("確率とIDの数が一致しません")

        self._probs = probabilities
        self._ids = strategy_ids

    @property
    def probabilities(self) -> np.ndarray:
        return self._probs

    @property
    def strategy_ids(self) -> list[str]:
        return self._ids

    def get_probability(self, strategy_id: str) -> float:
        """特定戦略の確率を取得"""
        try:
            idx = self._ids.index(strategy_id)
            return float(self._probs[idx])
        except ValueError:
            return 0.0

    def validate(self, tolerance: float = 1e-6) -> bool:
        """確率の合計が1か検証"""
        return abs(float(np.sum(self._probs)) - 1.0) < tolerance

    def get_support(self, threshold: float = 1e-6) -> list[str]:
        """サポート（正の確率を持つ戦略）を取得"""
        return [self._ids[i] for i, p in enumerate(self._probs) if p > threshold]

    def __repr__(self) -> str:
        return f"MixedStrategy(probabilities={self._probs}, strategy_ids={self._ids})"
