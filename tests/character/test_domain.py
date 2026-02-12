"""
character.domain モジュールのテスト

MatchupVector と Character クラスのテストを提供します。
"""

import pytest
import numpy as np
import random

from monocycle_nash.character.domain import MatchupVector, Character, get_characters


# =============================================================================
# MatchupVector のテスト
# =============================================================================

class TestMatchupVector:
    """MatchupVector クラスのテスト"""
    
    def test_init_with_xy(self):
        """x, y座標からの初期化"""
        v = MatchupVector(1.0, 2.0)
        assert v.x == 1.0
        assert v.y == 2.0
    
    def test_init_with_list(self):
        """リストからの初期化"""
        v = MatchupVector([1.0, 2.0])
        assert v.x == 1.0
        assert v.y == 2.0
    
    def test_init_with_array(self):
        """numpy配列からの初期化"""
        v = MatchupVector(np.array([1.0, 2.0]))
        assert v.x == 1.0
        assert v.y == 2.0
    
    def test_init_with_vector(self):
        """既存のMatchupVectorからの初期化（コピー）"""
        v1 = MatchupVector(1.0, 2.0)
        v2 = MatchupVector(v1)
        assert v2.x == 1.0
        assert v2.y == 2.0
        # コピーなので元の変更は影響しない
        v1._data[0] = 99.0
        assert v2.x == 1.0
    
    def test_init_invalid_shape(self):
        """無効な形状の場合はエラー"""
        with pytest.raises(ValueError, match="2次元ベクトル"):
            MatchupVector([1.0, 2.0, 3.0])
    
    def test_init_1d_scalar(self):
        """スカラーのみを渡すとエラー"""
        with pytest.raises(ValueError, match="2次元ベクトル"):
            MatchupVector(1.0)
    
    def test_times_basic(self):
        """外積計算の基本テスト"""
        v1 = MatchupVector(1.0, 0.0)
        v2 = MatchupVector(0.0, 1.0)
        assert v1.times(v2) == 1.0
        assert v2.times(v1) == -1.0
    
    def test_times_equilateral_triangle(self):
        """
        正三角形の外積計算テスト
        
        正三角形の頂点に対して、反時計回りに計算すると
        同じ値（2√3）になることを確認
        """
        ROOT3 = np.sqrt(3)
        v1 = MatchupVector(2.0, 0.0)
        v2 = MatchupVector(-1.0, ROOT3)
        v3 = MatchupVector(-1.0, -ROOT3)
        
        # 反時計回りの外積は正
        assert v1.times(v2) == pytest.approx(2 * ROOT3, rel=1e-4)
        assert v2.times(v3) == pytest.approx(2 * ROOT3, rel=1e-4)
        assert v3.times(v1) == pytest.approx(2 * ROOT3, rel=1e-4)
    
    def test_times_anticommutative(self):
        """外積の反交換性: a × b = - (b × a)"""
        for _ in range(10):
            x1, y1 = random.random(), random.random()
            x2, y2 = random.random(), random.random()
            v1 = MatchupVector(x1, y1)
            v2 = MatchupVector(x2, y2)
            
            assert v1.times(v2) == pytest.approx(-v2.times(v1))
    
    def test_times_self_zero(self):
        """自分自身との外積は0"""
        v = MatchupVector(random.random(), random.random())
        assert v.times(v) == 0.0
    
    def test_add(self):
        """ベクトル加算"""
        v1 = MatchupVector(1.0, 2.0)
        v2 = MatchupVector(3.0, 4.0)
        result = v1 + v2
        assert result.x == 4.0
        assert result.y == 6.0
    
    def test_sub(self):
        """ベクトル減算"""
        v1 = MatchupVector(5.0, 4.0)
        v2 = MatchupVector(3.0, 2.0)
        result = v1 - v2
        assert result.x == 2.0
        assert result.y == 2.0
    
    def test_mul_scalar(self):
        """スカラー乗算"""
        v = MatchupVector(1.0, 2.0)
        result = v * 3.0
        assert result.x == 3.0
        assert result.y == 6.0
    
    def test_rmul_scalar(self):
        """右からのスカラー乗算"""
        v = MatchupVector(1.0, 2.0)
        result = 3.0 * v
        assert result.x == 3.0
        assert result.y == 6.0
    
    def test_neg(self):
        """ベクトル否定"""
        v = MatchupVector(1.0, -2.0)
        result = -v
        assert result.x == -1.0
        assert result.y == 2.0
    
    def test_truediv_scalar(self):
        """スカラー除算"""
        v = MatchupVector(6.0, 4.0)
        result = v / 2.0
        assert result.x == 3.0
        assert result.y == 2.0
    
    def test_eq_with_vector(self):
        """MatchupVector同士の等価判定"""
        v1 = MatchupVector(1.0, 2.0)
        v2 = MatchupVector(1.0, 2.0)
        v3 = MatchupVector(1.0, 2.0000001)
        
        assert v1 == v2
        # 浮動小数点誤差を許容
        assert v1 == v3
    
    def test_eq_with_list(self):
        """リストとの等価判定"""
        v = MatchupVector(1.0, 2.0)
        assert v == [1.0, 2.0]
    
    def test_eq_with_array(self):
        """numpy配列との等価判定"""
        v = MatchupVector(1.0, 2.0)
        assert v == np.array([1.0, 2.0])
    
    def test_eq_invalid_type(self):
        """無効な型との比較はエラー"""
        v = MatchupVector(1.0, 2.0)
        with pytest.raises(TypeError):
            v == "invalid"
    
    def test_copy(self):
        """コピーメソッド"""
        v1 = MatchupVector(1.0, 2.0)
        v2 = v1.copy()
        
        assert v1 == v2
        # コピーなので独立
        v1._data[0] = 99.0
        assert v2.x == 1.0
    
    def test_to_array(self):
        """numpy配列への変換"""
        v = MatchupVector(1.0, 2.0)
        arr = v.to_array()
        
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (2,)
        assert arr[0] == 1.0
        assert arr[1] == 2.0
    
    def test_str(self):
        """文字列表現"""
        v = MatchupVector(1.0, 2.0)
        assert str(v) == "1.0,2.0"
    
    def test_repr(self):
        """repr表現"""
        v = MatchupVector(1.0, 2.0)
        assert repr(v) == "MatchupVector(1.0, 2.0)"


# =============================================================================
# Character のテスト
# =============================================================================

class TestCharacter:
    """Character クラスのテスト"""
    
    def test_init(self):
        """基本的な初期化"""
        v = MatchupVector(1.0, 2.0)
        c = Character(5.0, v)
        
        assert c.p == 5.0
        assert c.v.x == 1.0
        assert c.v.y == 2.0
    
    def test_init_copies_vector(self):
        """ベクトルはコピーされる"""
        v = MatchupVector(1.0, 2.0)
        c = Character(5.0, v)
        
        # 元のベクトルを変更しても影響しない
        v._data[0] = 99.0
        assert c.v.x == 1.0
    
    def test_tolist_default(self):
        """デフォルトのtolist（p, x, y順）"""
        c = Character(5.0, MatchupVector(1.0, 2.0))
        result = c.tolist()
        
        assert result == [5.0, 1.0, 2.0]
    
    def test_tolist_custom_order(self):
        """カスタム順序のtolist"""
        c = Character(5.0, MatchupVector(1.0, 2.0))
        
        assert c.tolist(["x", "y", "p"]) == [1.0, 2.0, 5.0]
        assert c.tolist(["y", "p"]) == [2.0, 5.0]
        assert c.tolist(["p"]) == [5.0]
    
    def test_tolist_invalid_key(self):
        """無効なキーはエラー"""
        c = Character(5.0, MatchupVector(1.0, 2.0))
        with pytest.raises(ValueError, match="無効なキー"):
            c.tolist(["invalid"])
    
    def test_convert_basic(self):
        """基本的な原点移動"""
        c = Character(5.0, MatchupVector(1.0, 2.0))
        a = MatchupVector(0.5, 0.5)
        
        new_c = c.convert(a)
        
        # p' = p + v × a
        expected_p = 5.0 + c.v.times(a)
        # v' = v - a
        expected_v = MatchupVector(0.5, 1.5)
        
        assert new_c.p == pytest.approx(expected_p)
        assert new_c.v == expected_v
    
    def test_convert_with_list(self):
        """リスト形式のaction_vectorでの原点移動"""
        c = Character(5.0, MatchupVector(1.0, 2.0))
        new_c = c.convert([0.5, 0.5])
        
        assert new_c.v.x == 0.5
        assert new_c.v.y == 1.5
    
    def test_convert_identity(self):
        """零ベクトルでの原点移動（恒等変換）"""
        c = Character(5.0, MatchupVector(1.0, 2.0))
        new_c = c.convert([0.0, 0.0])
        
        assert new_c.p == c.p
        assert new_c.v == c.v
    
    def test_str(self):
        """文字列表現"""
        c = Character(5.0, MatchupVector(1.0, 2.0))
        assert "power=5.0" in str(c)
        assert "vector=" in str(c)
    
    def test_repr(self):
        """repr表現"""
        c = Character(5.0, MatchupVector(1.0, 2.0))
        assert repr(c) == "Character(5.0, MatchupVector(1.0, 2.0))"
    
    def test_eq_same(self):
        """同じ値のキャラクターは等価"""
        c1 = Character(5.0, MatchupVector(1.0, 2.0))
        c2 = Character(5.0, MatchupVector(1.0, 2.0))
        
        assert c1 == c2
    
    def test_eq_different(self):
        """異なる値のキャラクターは非等価"""
        c1 = Character(5.0, MatchupVector(1.0, 2.0))
        c2 = Character(5.0, MatchupVector(1.0, 2.1))
        c3 = Character(5.1, MatchupVector(1.0, 2.0))
        
        assert c1 != c2
        assert c1 != c3
    
    def test_eq_non_character(self):
        """Character以外との比較はFalse"""
        c = Character(5.0, MatchupVector(1.0, 2.0))
        assert c != "not a character"
        assert c != 5.0
        assert c != None


# =============================================================================
# get_characters のテスト
# =============================================================================

class TestGetCharacters:
    """get_characters ヘルパー関数のテスト"""
    
    def test_get_characters_from_list(self):
        """リストからの生成"""
        data = [
            [1.0, 1.0, 0.0],
            [0.5, -0.5, 0.5],
            [0.5, -0.5, -0.5]
        ]
        
        chars = get_characters(data)
        
        assert len(chars) == 3
        assert chars[0].p == 1.0
        assert chars[0].v.x == 1.0
        assert chars[0].v.y == 0.0
        assert chars[1].p == 0.5
        assert chars[1].v.x == -0.5
        assert chars[1].v.y == 0.5
    
    def test_get_characters_from_array(self):
        """numpy配列からの生成"""
        data = np.array([
            [1.0, 1.0, 0.0],
            [0.5, -0.5, 0.5]
        ])
        
        chars = get_characters(data)
        
        assert len(chars) == 2
        assert chars[0].p == 1.0
    
    def test_get_characters_empty(self):
        """空リスト"""
        chars = get_characters([])
        assert chars == []
    
    def test_get_characters_single(self):
        """1要素のみ"""
        data = [[1.0, 2.0, 3.0]]
        chars = get_characters(data)
        
        assert len(chars) == 1
        assert chars[0].p == 1.0
        assert chars[0].v.x == 2.0
        assert chars[0].v.y == 3.0