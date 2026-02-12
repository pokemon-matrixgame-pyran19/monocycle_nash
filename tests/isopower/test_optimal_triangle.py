"""
OptimalTriangleFinderクラスのテスト

最適三角形探索機能のテスト。
"""

import pytest
import numpy as np
from monocycle_nash.isopower.triangle import OptimalTriangleFinder, OptimalTriangleResult
from monocycle_nash.matrix.monocycle import MonocyclePayoffMatrix
from monocycle_nash.character.domain import Character, MatchupVector
from tests.theory.builder import TheoryTestBuilder


ROOT3 = 1.7320508075688772


class TestOptimalTriangleFinder:
    """OptimalTriangleFinderクラスのテスト"""
    
    def test_find_best_returns_result_for_janken(self):
        """
        じゃんけん行列で最適三角形が見つかる
        """
        # じゃんけんのキャラクター（正三角形配置）
        characters = [
            Character(0.0, MatchupVector(2, 0)),
            Character(0.0, MatchupVector(-1, ROOT3)),
            Character(0.0, MatchupVector(-1, -ROOT3)),
        ]
        matrix = MonocyclePayoffMatrix(characters)
        
        finder = OptimalTriangleFinder(matrix)
        result = finder.find_best()
        
        # 結果が見つかる
        assert result is not None
        assert isinstance(result, OptimalTriangleResult)
    
    def test_find_best_returns_none_for_less_than_3(self):
        """
        キャラクターが3未満の場合はNoneを返す
        """
        characters = [
            Character(0.0, MatchupVector(1, 0)),
            Character(0.0, MatchupVector(0, 1)),
        ]
        matrix = MonocyclePayoffMatrix(characters)
        
        finder = OptimalTriangleFinder(matrix)
        result = finder.find_best()
        
        assert result is None
    
    def test_result_has_valid_indices(self):
        """
        結果のインデックスが有効
        """
        characters = [
            Character(0.0, MatchupVector(2, 0)),
            Character(0.0, MatchupVector(-1, ROOT3)),
            Character(0.0, MatchupVector(-1, -ROOT3)),
        ]
        matrix = MonocyclePayoffMatrix(characters)
        
        finder = OptimalTriangleFinder(matrix)
        result = finder.find_best()
        
        assert result is not None
        i, j, k = result.indices
        # インデックスが範囲内
        assert 0 <= i < len(characters)
        assert 0 <= j < len(characters)
        assert 0 <= k < len(characters)
        # インデックスが異なる
        assert i != j
        assert j != k
        assert k != i
    
    def test_result_has_valid_characters(self):
        """
        結果のキャラクターが有効
        """
        characters = [
            Character(0.0, MatchupVector(2, 0)),
            Character(0.0, MatchupVector(-1, ROOT3)),
            Character(0.0, MatchupVector(-1, -ROOT3)),
        ]
        matrix = MonocyclePayoffMatrix(characters)
        
        finder = OptimalTriangleFinder(matrix)
        result = finder.find_best()
        
        assert result is not None
        assert len(result.characters) == 3
        for c in result.characters:
            assert isinstance(c, Character)
    
    def test_result_has_valid_a_vector(self):
        """
        結果の等パワー座標aが有効
        """
        characters = [
            Character(0.0, MatchupVector(2, 0)),
            Character(0.0, MatchupVector(-1, ROOT3)),
            Character(0.0, MatchupVector(-1, -ROOT3)),
        ]
        matrix = MonocyclePayoffMatrix(characters)
        
        finder = OptimalTriangleFinder(matrix)
        result = finder.find_best()
        
        assert result is not None
        assert isinstance(result.a_vector, MatchupVector)
    
    def test_get_a_raises_when_not_found(self):
        """
        結果が見つからない場合、get_a()は例外を投げる
        """
        characters = [
            Character(0.0, MatchupVector(1, 0)),
            Character(0.0, MatchupVector(0, 1)),
        ]
        matrix = MonocyclePayoffMatrix(characters)
        
        finder = OptimalTriangleFinder(matrix)
        finder.find()
        
        with pytest.raises(ValueError):
            finder.get_a()
    
    def test_get_optimal_3characters_raises_when_not_found(self):
        """
        結果が見つからない場合、get_optimal_3characters()は例外を投げる
        """
        characters = [
            Character(0.0, MatchupVector(1, 0)),
            Character(0.0, MatchupVector(0, 1)),
        ]
        matrix = MonocyclePayoffMatrix(characters)
        
        finder = OptimalTriangleFinder(matrix)
        finder.find()
        
        with pytest.raises(ValueError):
            finder.get_optimal_3characters()
    
    def test_found_triangle_has_inner_a(self):
        """
        見つかった三角形のaは内部にある
        """
        # じゃんけんoriginal variant（等パワーでない）
        case = TheoryTestBuilder.janken()
        variant = case.variants[0]  # original
        
        characters = [
            Character(p, MatchupVector(v[0], v[1]))
            for p, v in zip(variant.powers, variant.vectors)
        ]
        matrix = MonocyclePayoffMatrix(characters)
        
        finder = OptimalTriangleFinder(matrix)
        result = finder.find_best()
        
        # 結果が見つかる（aが内部にある三角形が存在）
        assert result is not None
        
        # aが期待値と一致（テストビルダーのisopower_a）
        expected_a = variant.isopower_a
        assert result.a_vector.x == pytest.approx(expected_a[0], abs=1e-6)
        assert result.a_vector.y == pytest.approx(expected_a[1], abs=1e-6)
    
    def test_shifted_variant_finds_zero_a(self):
        """
        shifted variant（既に等パワー）ではa=(0,0)が見つかる
        """
        case = TheoryTestBuilder.janken()
        variant = case.variants[1]  # shifted
        
        characters = [
            Character(p, MatchupVector(v[0], v[1]))
            for p, v in zip(variant.powers, variant.vectors)
        ]
        matrix = MonocyclePayoffMatrix(characters)
        
        finder = OptimalTriangleFinder(matrix)
        result = finder.find_best()
        
        # 結果が見つかる
        assert result is not None
        # aは(0, 0)に近い
        assert result.a_vector.x == pytest.approx(0.0, abs=1e-6)
        assert result.a_vector.y == pytest.approx(0.0, abs=1e-6)
    
    def test_get_result_returns_list(self):
        """
        get_result()はリストを返す
        """
        characters = [
            Character(0.0, MatchupVector(2, 0)),
            Character(0.0, MatchupVector(-1, ROOT3)),
            Character(0.0, MatchupVector(-1, -ROOT3)),
        ]
        matrix = MonocyclePayoffMatrix(characters)
        
        finder = OptimalTriangleFinder(matrix)
        finder.find()
        
        results = finder.get_result()
        assert isinstance(results, list)
        assert len(results) == 1
        assert isinstance(results[0], OptimalTriangleResult)
