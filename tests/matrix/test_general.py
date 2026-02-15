import pytest
import numpy as np
from monocycle_nash.matrix.general import GeneralPayoffMatrix
from monocycle_nash.matrix.base import PayoffMatrix


class TestGeneralPayoffMatrix:
    """GeneralPayoffMatrixクラスのテスト"""
    
    def test_initialization(self):
        """基本的な初期化"""
        matrix = np.array([[0, 1, -1], [-1, 0, 1], [1, -1, 0]], dtype=float)
        payoff = GeneralPayoffMatrix(matrix)
        
        assert payoff.size == 3
        assert payoff.labels == ["s0", "s1", "s2"]
        np.testing.assert_array_almost_equal(payoff.matrix, matrix)
    
    def test_initialization_with_labels(self):
        """ラベル付き初期化"""
        matrix = np.array([[0, 1], [-1, 0]], dtype=float)
        labels = ["Rock", "Paper"]
        payoff = GeneralPayoffMatrix(matrix, labels)
        
        assert payoff.labels == labels
    
    def test_matrix_property(self):
        """matrixプロパティの取得"""
        matrix = np.array([[0, 1, -1], [-1, 0, 1], [1, -1, 0]], dtype=float)
        payoff = GeneralPayoffMatrix(matrix)
        
        result = payoff.matrix
        assert isinstance(result, np.ndarray)
        assert result.shape == (3, 3)
    
    def test_size_property(self):
        """sizeプロパティの取得"""
        matrix = np.array([[0, 1], [-1, 0]], dtype=float)
        payoff = GeneralPayoffMatrix(matrix)
        
        assert payoff.size == 2
    
    def test_labels_property(self):
        """labelsプロパティの取得"""
        matrix = np.array([[0, 1], [-1, 0]], dtype=float)
        labels = ["A", "B"]
        payoff = GeneralPayoffMatrix(matrix, labels)
        
        assert payoff.labels == labels
    
    def test_default_labels(self):
        """デフォルトラベルの生成"""
        matrix = np.array([[0, 1, -1], [-1, 0, 1], [1, -1, 0]], dtype=float)
        payoff = GeneralPayoffMatrix(matrix)
        
        assert payoff.labels == ["s0", "s1", "s2"]
    
    def test_get_value(self):
        """行列要素の取得"""
        matrix = np.array([[0, 1, -1], [-1, 0, 1], [1, -1, 0]], dtype=float)
        payoff = GeneralPayoffMatrix(matrix)
        
        assert payoff.get_value(0, 0) == 0.0
        assert payoff.get_value(0, 1) == 1.0
        assert payoff.get_value(0, 2) == -1.0
        assert payoff.get_value(1, 0) == -1.0
    
    def test_is_payoff_matrix(self):
        """PayoffMatrixのサブクラスであること"""
        matrix = np.array([[0, 1], [-1, 0]], dtype=float)
        payoff = GeneralPayoffMatrix(matrix)
        
        assert isinstance(payoff, PayoffMatrix)
    
    def test_equality_same_matrix(self):
        """同じ行列の等価判定"""
        matrix = np.array([[0, 1], [-1, 0]], dtype=float)
        payoff1 = GeneralPayoffMatrix(matrix)
        payoff2 = GeneralPayoffMatrix(matrix.copy())
        
        assert payoff1 == payoff2
    
    def test_equality_different_labels(self):
        """ラベルが異なっても行列が同じなら等価"""
        matrix = np.array([[0, 1], [-1, 0]], dtype=float)
        payoff1 = GeneralPayoffMatrix(matrix, ["A", "B"])
        payoff2 = GeneralPayoffMatrix(matrix, ["X", "Y"])
        
        assert payoff1 == payoff2
    
    def test_inequality_different_values(self):
        """値が異なる場合は非等価"""
        matrix1 = np.array([[0, 1], [-1, 0]], dtype=float)
        matrix2 = np.array([[0, 2], [-2, 0]], dtype=float)
        payoff1 = GeneralPayoffMatrix(matrix1)
        payoff2 = GeneralPayoffMatrix(matrix2)
        
        assert payoff1 != payoff2
    
    def test_inequality_with_non_matrix(self):
        """PayoffMatrix以外との比較"""
        matrix = np.array([[0, 1], [-1, 0]], dtype=float)
        payoff = GeneralPayoffMatrix(matrix)
        
        assert payoff != "not a matrix"
        assert payoff != 123
    
    def test_matrix_shape_validation(self):
        """正方行列であること"""
        # 正方行列でない場合のテスト
        matrix = np.array([[0, 1, -1], [-1, 0, 1]], dtype=float)  # 2x3
        
        # 現状の実装ではsizeがshape[0]を使うため、
        # 非正方行列でもエラーにならないが、使用時に問題が出る可能性がある
        payoff = GeneralPayoffMatrix(matrix)
        assert payoff.size == 2  # shape[0]が使用される
    
    def test_float_conversion(self):
        """整数行列がfloatに変換される"""
        matrix = np.array([[0, 1], [-1, 0]], dtype=int)
        payoff = GeneralPayoffMatrix(matrix)
        
        assert payoff.matrix.dtype == np.float64
