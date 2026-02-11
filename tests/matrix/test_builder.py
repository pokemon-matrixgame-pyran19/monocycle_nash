import pytest
import numpy as np
from matrix.builder import PayoffMatrixBuilder
from matrix.general import GeneralPayoffMatrix
from matrix.monocycle import MonocyclePayoffMatrix
from rule.character import Character, MatchupVector


class TestPayoffMatrixBuilder:
    """PayoffMatrixBuilderクラスのテスト"""
    
    def test_from_characters(self):
        """CharacterリストからMonocyclePayoffMatrix生成"""
        characters = [
            Character(1.0, MatchupVector(1, 0)),
            Character(0.5, MatchupVector(-0.5, 0.5)),
            Character(0.5, MatchupVector(-0.5, -0.5)),
        ]
        
        matrix = PayoffMatrixBuilder.from_characters(characters)
        
        assert isinstance(matrix, MonocyclePayoffMatrix)
        assert matrix.size == 3
    
    def test_from_characters_with_labels(self):
        """Characterリストからラベル付きで生成"""
        characters = [
            Character(1.0, MatchupVector(1, 0)),
            Character(0.5, MatchupVector(-0.5, 0.5)),
        ]
        labels = ["Fast", "Slow"]
        
        matrix = PayoffMatrixBuilder.from_characters(characters, labels)
        
        assert matrix.labels == labels
    
    def test_from_general_matrix(self):
        """任意行列からGeneralPayoffMatrix生成"""
        data = np.array([[0, 1, -1], [-1, 0, 1], [1, -1, 0]], dtype=float)
        
        matrix = PayoffMatrixBuilder.from_general_matrix(data)
        
        assert isinstance(matrix, GeneralPayoffMatrix)
        assert matrix.size == 3
        np.testing.assert_array_almost_equal(matrix.matrix, data)
    
    def test_from_general_matrix_with_labels(self):
        """任意行列からラベル付きで生成"""
        data = np.array([[0, 1], [-1, 0]], dtype=float)
        labels = ["Rock", "Paper"]
        
        matrix = PayoffMatrixBuilder.from_general_matrix(data, labels)
        
        assert matrix.labels == labels
    
    def test_builder_creates_different_types(self):
        """Builderが異なる型を生成できる"""
        # MonocyclePayoffMatrix
        characters = [
            Character(1.0, MatchupVector(1, 0)),
            Character(0.5, MatchupVector(0, 1)),
        ]
        mono_matrix = PayoffMatrixBuilder.from_characters(characters)
        
        # GeneralPayoffMatrix
        data = np.array([[0, 1], [-1, 0]], dtype=float)
        gen_matrix = PayoffMatrixBuilder.from_general_matrix(data)
        
        assert isinstance(mono_matrix, MonocyclePayoffMatrix)
        assert isinstance(gen_matrix, GeneralPayoffMatrix)
        assert type(mono_matrix) != type(gen_matrix)
    
    def test_from_characters_preserves_character_data(self):
        """Characterデータが保持される"""
        characters = [
            Character(1.5, MatchupVector(2, 3)),
            Character(0.5, MatchupVector(-1, -2)),
        ]
        
        matrix = PayoffMatrixBuilder.from_characters(characters)
        
        assert matrix.characters[0].p == 1.5
        assert matrix.characters[0].v.x == 2.0
        assert matrix.characters[0].v.y == 3.0
        assert matrix.characters[1].p == 0.5
        assert matrix.characters[1].v.x == -1.0
        assert matrix.characters[1].v.y == -2.0
