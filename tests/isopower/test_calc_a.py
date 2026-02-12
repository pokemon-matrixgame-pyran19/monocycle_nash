"""
aCalculatorクラスのテスト

テストビルダー（tests/theory/builder.py）を参照して、
理論的に正しい等パワー座標aが計算できることを検証します。
"""

import pytest
from monocycle_nash.isopower.calc_a import aCalculator
from monocycle_nash.character.domain import Character, MatchupVector
from tests.theory.builder import TheoryTestBuilder


class TestACalculator:
    """aCalculatorクラスのテスト"""
    
    def test_janken_variant1_isopower_a(self):
        """
        じゃんけんvariant1（original）の等パワー座標aを計算
        
        テストビルダーのデータを使用し、期待値と一致することを検証。
        """
        # テストビルダーからじゃんけんケースを取得
        case = TheoryTestBuilder.janken()
        variant = case.variants[0]  # original variant
        
        # Characterリストを作成
        characters = [
            Character(p, MatchupVector(v[0], v[1]))
            for p, v in zip(variant.powers, variant.vectors)
        ]
        
        # 等パワー座標aを計算
        calc = aCalculator(characters[0], characters[1], characters[2])
        a = calc.calc()
        
        # 期待値を取得（テストビルダーから）
        expected = variant.isopower_a
        
        # 検証
        assert expected is not None, "テストビルダーにisopower_aが設定されていません"
        assert a.x == pytest.approx(expected[0], abs=1e-6)
        assert a.y == pytest.approx(expected[1], abs=1e-6)
    
    def test_janken_variant1_is_inner(self):
        """
        じゃんけんvariant1でaが三角形の内部にあることを確認
        """
        case = TheoryTestBuilder.janken()
        variant = case.variants[0]
        
        characters = [
            Character(p, MatchupVector(v[0], v[1]))
            for p, v in zip(variant.powers, variant.vectors)
        ]
        
        calc = aCalculator(characters[0], characters[1], characters[2])
        a = calc.calc()
        
        # aが三角形の内部にあるはず
        assert calc.is_inner == True, f"a={a}は三角形の内部にありません"
    
    def test_janken_variant2_already_isopower(self):
        """
        じゃんけんvariant2（shifted）は既に等パワー（a=(0,0)）
        """
        case = TheoryTestBuilder.janken()
        variant = case.variants[1]  # shifted variant
        
        characters = [
            Character(p, MatchupVector(v[0], v[1]))
            for p, v in zip(variant.powers, variant.vectors)
        ]
        
        calc = aCalculator(characters[0], characters[1], characters[2])
        a = calc.calc()
        
        # 既に等パワーなのでaは(0, 0)のはず
        expected = variant.isopower_a
        assert expected is not None
        assert a.x == pytest.approx(expected[0], abs=1e-6)
        assert a.y == pytest.approx(expected[1], abs=1e-6)
    
    def test_calculation_returns_matchup_vector(self):
        """
        calc()がMatchupVectorを返す
        """
        case = TheoryTestBuilder.janken()
        variant = case.variants[0]
        
        characters = [
            Character(p, MatchupVector(v[0], v[1]))
            for p, v in zip(variant.powers, variant.vectors)
        ]
        
        calc = aCalculator(characters[0], characters[1], characters[2])
        a = calc.calc()
        
        assert isinstance(a, MatchupVector)
    
    def test_is_edge_property_exists(self):
        """
        is_edgeプロパティが存在する
        """
        case = TheoryTestBuilder.janken()
        variant = case.variants[0]
        
        characters = [
            Character(p, MatchupVector(v[0], v[1]))
            for p, v in zip(variant.powers, variant.vectors)
        ]
        
        calc = aCalculator(characters[0], characters[1], characters[2])
        calc.calc()
        
        # is_edgeプロパティにアクセスできる
        _ = calc.is_edge
