import pytest
import numpy as np
from monocycle_nash.equilibrium.domain import MixedStrategy


class TestMixedStrategy:
    """MixedStrategyクラスのテスト"""
    
    def test_initialization(self):
        """基本的な初期化"""
        probs = np.array([0.5, 0.3, 0.2])
        ids = ["A", "B", "C"]
        strategy = MixedStrategy(probs, ids)
        
        assert strategy.strategy_ids == ids
        np.testing.assert_array_almost_equal(strategy.probabilities, probs)
    
    def test_initialization_mismatch(self):
        """確率とIDの数が不一致の場合はエラー"""
        probs = np.array([0.5, 0.5])
        ids = ["A", "B", "C"]
        
        with pytest.raises(ValueError, match="確率とIDの数が一致しません"):
            MixedStrategy(probs, ids)
    
    def test_get_probability(self):
        """特定戦略の確率取得"""
        probs = np.array([0.5, 0.3, 0.2])
        ids = ["A", "B", "C"]
        strategy = MixedStrategy(probs, ids)
        
        assert strategy.get_probability("A") == pytest.approx(0.5)
        assert strategy.get_probability("B") == pytest.approx(0.3)
        assert strategy.get_probability("C") == pytest.approx(0.2)
        assert strategy.get_probability("D") == 0.0  # 存在しないID
    
    def test_validate_success(self):
        """確率の合計が1の場合は検証成功"""
        probs = np.array([0.5, 0.3, 0.2])
        ids = ["A", "B", "C"]
        strategy = MixedStrategy(probs, ids)
        
        assert strategy.validate() is True
        assert strategy.validate(tolerance=1e-10) is True
    
    def test_validate_failure(self):
        """確率の合計が1でない場合は検証失敗"""
        probs = np.array([0.5, 0.3, 0.1])  # 合計0.9
        ids = ["A", "B", "C"]
        strategy = MixedStrategy(probs, ids)
        
        assert strategy.validate() is False
    
    def test_validate_with_tolerance(self):
        """許容範囲内の誤差は許容"""
        probs = np.array([0.333, 0.333, 0.334])  # 合計1.0（ほぼ）
        ids = ["A", "B", "C"]
        strategy = MixedStrategy(probs, ids)
        
        assert strategy.validate(tolerance=1e-2) is True
    
    def test_get_support(self):
        """サポート（正の確率を持つ戦略）を取得"""
        probs = np.array([0.5, 0.0, 0.5])
        ids = ["A", "B", "C"]
        strategy = MixedStrategy(probs, ids)
        
        support = strategy.get_support()
        assert "A" in support
        assert "B" not in support
        assert "C" in support
    
    def test_get_support_with_threshold(self):
        """閾値を設定してサポートを取得"""
        probs = np.array([0.5, 0.0001, 0.4999])
        ids = ["A", "B", "C"]
        strategy = MixedStrategy(probs, ids)
        
        support = strategy.get_support(threshold=1e-3)
        assert "A" in support
        assert "B" not in support  # 閾値以下
        assert "C" in support
    
    def test_equality(self):
        """等価判定"""
        probs1 = np.array([0.5, 0.3, 0.2])
        probs2 = np.array([0.5, 0.3, 0.2])
        ids = ["A", "B", "C"]
        
        strategy1 = MixedStrategy(probs1, ids)
        strategy2 = MixedStrategy(probs2, ids)
        
        assert strategy1 == strategy2
    
    def test_inequality_different_ids(self):
        """IDが異なる場合は非等価"""
        probs = np.array([0.5, 0.3, 0.2])
        
        strategy1 = MixedStrategy(probs, ["A", "B", "C"])
        strategy2 = MixedStrategy(probs, ["X", "Y", "Z"])
        
        assert strategy1 != strategy2
    
    def test_inequality_different_probs(self):
        """確率が異なる場合は非等価"""
        ids = ["A", "B", "C"]
        
        strategy1 = MixedStrategy(np.array([0.5, 0.3, 0.2]), ids)
        strategy2 = MixedStrategy(np.array([0.6, 0.2, 0.2]), ids)
        
        assert strategy1 != strategy2
    
    def test_inequality_with_non_mixedstrategy(self):
        """MixedStrategy以外との比較"""
        probs = np.array([0.5, 0.3, 0.2])
        ids = ["A", "B", "C"]
        strategy = MixedStrategy(probs, ids)
        
        assert strategy != "not a strategy"
        assert strategy != 123
        assert strategy != None
    
    def test_repr(self):
        """文字列表現"""
        probs = np.array([0.5, 0.3, 0.2])
        ids = ["A", "B", "C"]
        strategy = MixedStrategy(probs, ids)
        
        repr_str = repr(strategy)
        assert "MixedStrategy" in repr_str
        assert "A:" in repr_str
        assert "B:" in repr_str
        assert "C:" in repr_str
