"""
calculator モジュールのテスト

PowerVectorCalculator クラスのテストを提供します。
"""

import pytest
import numpy as np

from monocycle_nash.character.calculator import PowerVectorCalculator
from monocycle_nash.character.domain import MatchupVector, Character, get_characters


class TestPowerVectorCalculator:
    """PowerVectorCalculator クラスのテスト"""
    
    @pytest.fixture
    def sample_characters(self):
        """テスト用のサンプルキャラクターリスト"""
        return [
            Character(5.0, MatchupVector(1.0, 0.0)),
            Character(3.0, MatchupVector(0.0, 1.0)),
            Character(4.0, MatchupVector(-1.0, 0.0)),
        ]
    
    @pytest.fixture
    def single_character(self):
        """1つのキャラクターのリスト"""
        return [Character(1.0, MatchupVector(2.0, 3.0))]
    
    def test_get_power_vector(self, sample_characters):
        """パワーベクトルの取得"""
        powers = PowerVectorCalculator.get_power_vector(sample_characters)
        
        assert isinstance(powers, np.ndarray)
        assert powers.shape == (3,)
        assert powers[0] == 5.0
        assert powers[1] == 3.0
        assert powers[2] == 4.0
    
    def test_get_power_vector_empty(self):
        """空リストのパワーベクトル"""
        powers = PowerVectorCalculator.get_power_vector([])
        
        assert isinstance(powers, np.ndarray)
        assert powers.shape == (0,)
    
    def test_get_matchup_vectors(self, sample_characters):
        """相性ベクトル行列の取得"""
        vectors = PowerVectorCalculator.get_matchup_vectors(sample_characters)
        
        assert isinstance(vectors, np.ndarray)
        assert vectors.shape == (3, 2)
        assert vectors[0, 0] == 1.0
        assert vectors[0, 1] == 0.0
        assert vectors[1, 0] == 0.0
        assert vectors[1, 1] == 1.0
        assert vectors[2, 0] == -1.0
        assert vectors[2, 1] == 0.0
    
    def test_get_matchup_vectors_empty(self):
        """空リストの相性ベクトル"""
        vectors = PowerVectorCalculator.get_matchup_vectors([])
        
        assert isinstance(vectors, np.ndarray)
        assert vectors.shape == (0, 2)
    
    def test_calculate_power_stats(self, sample_characters):
        """パワー統計の計算"""
        stats = PowerVectorCalculator.calculate_power_stats(sample_characters)
        
        assert stats["mean"] == pytest.approx(4.0)  # (5+3+4)/3
        assert stats["min"] == 3.0
        assert stats["max"] == 5.0
        assert stats["std"] > 0.0
    
    def test_calculate_power_stats_single(self, single_character):
        """1キャラクターの統計"""
        stats = PowerVectorCalculator.calculate_power_stats(single_character)
        
        assert stats["mean"] == 1.0
        assert stats["min"] == 1.0
        assert stats["max"] == 1.0
        assert stats["std"] == 0.0
    
    def test_calculate_power_stats_empty(self):
        """空リストの統計はエラー"""
        with pytest.raises(ZeroDivisionError):
            PowerVectorCalculator.calculate_power_stats([])
    
    def test_find_min_power_index(self, sample_characters):
        """最小パワーのインデックス"""
        idx = PowerVectorCalculator.find_min_power_index(sample_characters)
        
        assert idx == 1  # 3.0が最小
    
    def test_find_min_power_index_empty(self):
        """空リストの最小インデックスはエラー"""
        with pytest.raises(ValueError):
            PowerVectorCalculator.find_min_power_index([])
    
    def test_find_max_power_index(self, sample_characters):
        """最大パワーのインデックス"""
        idx = PowerVectorCalculator.find_max_power_index(sample_characters)
        
        assert idx == 0  # 5.0が最大
    
    def test_find_max_power_index_empty(self):
        """空リストの最大インデックスはエラー"""
        with pytest.raises(ValueError):
            PowerVectorCalculator.find_max_power_index([])
    
    def test_calculate_centroid(self, sample_characters):
        """重心の計算"""
        centroid = PowerVectorCalculator.calculate_centroid(sample_characters)
        
        assert isinstance(centroid, MatchupVector)
        assert centroid.x == pytest.approx(0.0)  # (1+0-1)/3
        assert centroid.y == pytest.approx(1.0/3.0)  # (0+1+0)/3
    
    def test_calculate_centroid_empty(self):
        """空リストの重心はエラー"""
        with pytest.raises(ValueError, match="キャラクターリストが空"):
            PowerVectorCalculator.calculate_centroid([])
    
    def test_calculate_centroid_single(self, single_character):
        """1キャラクターの重心"""
        centroid = PowerVectorCalculator.calculate_centroid(single_character)
        
        assert centroid.x == 2.0
        assert centroid.y == 3.0
    
    def test_shift_to_origin_with_default(self, sample_characters):
        """デフォルト（重心）での原点移動"""
        shifted = PowerVectorCalculator.shift_to_origin(sample_characters)
        
        assert len(shifted) == 3
        
        # 重心を原点に移動したので、新しい重心は(0, 0)付近
        new_centroid = PowerVectorCalculator.calculate_centroid(shifted)
        assert abs(new_centroid.x) < 1e-10
        assert abs(new_centroid.y) < 1e-10
    
    def test_shift_to_origin_with_custom(self, sample_characters):
        """カスタム原点での移動"""
        origin = MatchupVector(1.0, 0.0)
        shifted = PowerVectorCalculator.shift_to_origin(sample_characters, origin)
        
        assert len(shifted) == 3
        
        # 1番目のキャラクター: (1,0) → (0,0)
        assert shifted[0].v.x == pytest.approx(0.0)
        assert shifted[0].v.y == pytest.approx(0.0)
        
        # 2番目のキャラクター: (0,1) → (-1,1)
        assert shifted[1].v.x == pytest.approx(-1.0)
        assert shifted[1].v.y == pytest.approx(1.0)
    
    def test_shift_to_origin_with_list(self, sample_characters):
        """リスト形式の原点指定"""
        shifted = PowerVectorCalculator.shift_to_origin(sample_characters, [1.0, 0.0])
        
        assert len(shifted) == 3
        assert shifted[0].v.x == pytest.approx(0.0)
    
    def test_shift_to_origin_empty(self):
        """空リストの原点移動"""
        shifted = PowerVectorCalculator.shift_to_origin([])
        assert shifted == []


class TestIntegrationWithGetCharacters:
    """get_charactersとの統合テスト"""
    
    def test_end_to_end_workflow(self):
        """エンドツーエンドのワークフロー"""
        # 1. データからキャラクターを生成
        data = [
            [1.0, 1.0, 0.0],
            [0.5, -0.5, 0.5],
            [0.5, -0.5, -0.5],
        ]
        characters = get_characters(data)
        
        # 2. パワー統計を計算
        stats = PowerVectorCalculator.calculate_power_stats(characters)
        assert stats["mean"] == pytest.approx(2.0/3.0)
        
        # 3. 重心を計算
        centroid = PowerVectorCalculator.calculate_centroid(characters)
        assert centroid.x == pytest.approx(0.0)
        
        # 4. 原点移動
        shifted = PowerVectorCalculator.shift_to_origin(characters)
        new_centroid = PowerVectorCalculator.calculate_centroid(shifted)
        assert abs(new_centroid.x) < 1e-10
        assert abs(new_centroid.y) < 1e-10