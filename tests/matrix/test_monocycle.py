import pytest
import numpy as np
from monocycle_nash.matrix.monocycle import MonocyclePayoffMatrix
from monocycle_nash.matrix.base import PayoffMatrix
from monocycle_nash.character.domain import Character, MatchupVector


ROOT3 = 1.7320508075688772


class TestMonocyclePayoffMatrix:
    """MonocyclePayoffMatrixクラスのテスト"""
    
    def test_initialization(self):
        """基本的な初期化"""
        characters = [
            Character(1.0, MatchupVector(1, 0)),
            Character(0.5, MatchupVector(-0.5, 0.5)),
            Character(0.5, MatchupVector(-0.5, -0.5)),
        ]
        payoff = MonocyclePayoffMatrix(characters)
        
        assert payoff.size == 3
        assert payoff.labels == ["c0", "c1", "c2"]
        assert len(payoff.characters) == 3
    
    def test_initialization_with_labels(self):
        """ラベル付き初期化"""
        characters = [
            Character(1.0, MatchupVector(1, 0)),
            Character(0.5, MatchupVector(-0.5, 0.5)),
        ]
        labels = ["Fast", "Slow"]
        payoff = MonocyclePayoffMatrix(characters, labels)
        
        assert payoff.labels == labels
    
    def test_characters_property(self):
        """charactersプロパティの取得"""
        characters = [
            Character(1.0, MatchupVector(1, 0)),
            Character(0.5, MatchupVector(-0.5, 0.5)),
        ]
        payoff = MonocyclePayoffMatrix(characters)
        
        result = payoff.characters
        assert result == characters
        assert result[0].p == 1.0
        assert result[1].p == 0.5
    
    def test_matrix_calculation(self):
        """Aij = pi - pj + vi×vj の計算"""
        # 正三角形の設定
        power = 1.0
        characters = [
            Character(power, MatchupVector(2, 0)),
            Character(power, MatchupVector(-1, ROOT3)),
            Character(power + 1, MatchupVector(-1, -ROOT3)),
        ]
        payoff = MonocyclePayoffMatrix(characters)
        
        matrix = payoff.matrix
        
        # 形状の確認
        assert matrix.shape == (3, 3)
        
        # 対角成分は0（vi×vi = 0, pi - pi = 0）
        assert matrix[0, 0] == pytest.approx(0.0, abs=1e-10)
        assert matrix[1, 1] == pytest.approx(0.0, abs=1e-10)
        assert matrix[2, 2] == pytest.approx(0.0, abs=1e-10)
    
    def test_matrix_calculation_values(self):
        """行列値の具体的な計算"""
        # シンプルな2キャラクター
        characters = [
            Character(1.0, MatchupVector(1, 0)),
            Character(0.5, MatchupVector(0, 1)),
        ]
        payoff = MonocyclePayoffMatrix(characters)
        
        matrix = payoff.matrix
        
        # A[0,1] = p0 - p1 + v0×v1 = 1.0 - 0.5 + (1*1 - 0*0) = 0.5 + 1 = 1.5
        assert matrix[0, 1] == pytest.approx(1.5, abs=1e-10)
        
        # A[1,0] = p1 - p0 + v1×v0 = 0.5 - 1.0 + (0*0 - 1*1) = -0.5 - 1 = -1.5
        assert matrix[1, 0] == pytest.approx(-1.5, abs=1e-10)
    
    def test_get_power_vector(self):
        """パワーベクトルの取得"""
        characters = [
            Character(1.0, MatchupVector(1, 0)),
            Character(0.5, MatchupVector(-0.5, 0.5)),
            Character(0.0, MatchupVector(-0.5, -0.5)),
        ]
        payoff = MonocyclePayoffMatrix(characters)
        
        power_vector = payoff.get_power_vector()
        
        expected = np.array([1.0, 0.5, 0.0])
        np.testing.assert_array_almost_equal(power_vector, expected)
    
    def test_get_matchup_vectors(self):
        """相性ベクトルの取得"""
        characters = [
            Character(1.0, MatchupVector(1, 2)),
            Character(0.5, MatchupVector(3, 4)),
        ]
        payoff = MonocyclePayoffMatrix(characters)
        
        matchup_vectors = payoff.get_matchup_vectors()
        
        expected = np.array([[1, 2], [3, 4]])
        np.testing.assert_array_almost_equal(matchup_vectors, expected)
    
    def test_shift_origin(self):
        """等パワー座標での原点移動"""
        characters = [
            Character(1.0, MatchupVector(2, 0)),
            Character(0.5, MatchupVector(-1, 1)),
            Character(0.0, MatchupVector(-1, -1)),
        ]
        payoff = MonocyclePayoffMatrix(characters)
        
        # 原点移動前の行列を保存
        original_matrix = payoff.matrix.copy()
        
        # 原点移動
        a = MatchupVector(1, 0)
        shifted = payoff.shift_origin(a)
        
        # 移動後も利得行列は同じ（等価性保持）
        np.testing.assert_array_almost_equal(shifted.matrix, original_matrix)
    
    def test_shift_origin_properties(self):
        """原点移動後のプロパティ"""
        characters = [
            Character(1.0, MatchupVector(2, 0)),
            Character(0.5, MatchupVector(-1, 1)),
        ]
        labels = ["A", "B"]
        payoff = MonocyclePayoffMatrix(characters, labels)
        
        a = MatchupVector(1, 0)
        shifted = payoff.shift_origin(a)
        
        # ラベルは保持される
        assert shifted.labels == labels
        
        # サイズは同じ
        assert shifted.size == payoff.size
    
    def test_is_payoff_matrix(self):
        """PayoffMatrixのサブクラスであること"""
        characters = [
            Character(1.0, MatchupVector(1, 0)),
        ]
        payoff = MonocyclePayoffMatrix(characters)
        
        assert isinstance(payoff, PayoffMatrix)
    
    def test_equality_same_characters(self):
        """同じキャラクターでの等価判定"""
        characters = [
            Character(1.0, MatchupVector(1, 0)),
            Character(0.5, MatchupVector(0, 1)),
        ]
        payoff1 = MonocyclePayoffMatrix(characters)
        payoff2 = MonocyclePayoffMatrix(characters)
        
        assert payoff1 == payoff2
    
    def test_equality_rotated(self):
        """回転したキャラクターでも同じ利得行列なら等価"""
        # 回転前
        characters1 = [
            Character(1.0, MatchupVector(2, 0)),
            Character(1.0, MatchupVector(-1, ROOT3)),
            Character(1.0, MatchupVector(-1, -ROOT3)),
        ]
        
        # 90度回転
        characters2 = [
            Character(1.0, MatchupVector(0, 2)),
            Character(1.0, MatchupVector(-ROOT3, -1)),
            Character(1.0, MatchupVector(ROOT3, -1)),
        ]
        
        payoff1 = MonocyclePayoffMatrix(characters1)
        payoff2 = MonocyclePayoffMatrix(characters2)
        
        # 同じパワーで回転のみなので利得行列は同じ
        assert payoff1 == payoff2
    
    def test_get_value(self):
        """行列要素の取得（基底クラスのメソッド）"""
        characters = [
            Character(1.0, MatchupVector(1, 0)),
            Character(0.5, MatchupVector(0, 1)),
        ]
        payoff = MonocyclePayoffMatrix(characters)
        
        assert payoff.get_value(0, 0) == pytest.approx(0.0, abs=1e-10)
        assert payoff.get_value(0, 1) == pytest.approx(1.5, abs=1e-10)
