"""
aCalculatorクラスのテスト

テストビルダー（tests/theory/builder.py）を参照して、
理論的に正しい等パワー座標aが計算できることを検証します。
"""

import pytest
from monocycle_nash.isopower.calc_a import aCalculator
from monocycle_nash.character.domain import Character, MatchupVector
from theory.builder import TheoryTestBuilder


class TestACalculator:
    """aCalculatorクラスのテスト"""
    
    def test_all_cases_isopower_a(self):
        """
        全テストケースの全variantで等パワー座標aを計算・検証
        
        テストビルダーのget_all_cases()をループし、
        isopower_aが設定されているvariantで計算結果を検証。
        """
        all_cases = TheoryTestBuilder.get_all_cases()
        
        for case in all_cases:
            for variant in case.variants:
                # isopower_aが設定されていないvariantはスキップ
                if variant.isopower_a is None:
                    continue
                
                # 最低3キャラクター必要
                if len(variant.powers) < 3 or len(variant.vectors) < 3:
                    continue
                
                # Characterリストを作成
                characters = [
                    Character(p, MatchupVector(v[0], v[1]))
                    for p, v in zip(variant.powers, variant.vectors)
                ]
                
                # 等パワー座標aを計算
                calc = aCalculator(characters[0], characters[1], characters[2])
                a = calc.calc()
                
                # 期待値と比較
                expected = variant.isopower_a
                assert a.x == pytest.approx(expected[0], abs=1e-6), \
                    f"{case.name}/{variant.name}: x座標が一致しません (期待: {expected[0]}, 実際: {a.x})"
                assert a.y == pytest.approx(expected[1], abs=1e-6), \
                    f"{case.name}/{variant.name}: y座標が一致しません (期待: {expected[1]}, 実際: {a.y})"
    
    def test_all_cases_is_inner(self):
        """
        全テストケースでaが三角形の内部にあることを確認
        """
        all_cases = TheoryTestBuilder.get_all_cases()
        
        for case in all_cases:
            for variant in case.variants:
                if variant.isopower_a is None:
                    continue
                
                if len(variant.powers) < 3 or len(variant.vectors) < 3:
                    continue
                
                characters = [
                    Character(p, MatchupVector(v[0], v[1]))
                    for p, v in zip(variant.powers, variant.vectors)
                ]
                
                calc = aCalculator(characters[0], characters[1], characters[2])
                a = calc.calc()
                
                # aが三角形の内部にあるはず
                assert calc.is_inner == True, \
                    f"{case.name}/{variant.name}: a={a}は三角形の内部にありません"
    
    def test_calculation_returns_matchup_vector(self):
        """
        calc()がMatchupVectorを返す
        
        全ケースの最初の有効なvariantで検証。
        """
        all_cases = TheoryTestBuilder.get_all_cases()
        
        # 最初の有効なvariantを見つける
        for case in all_cases:
            for variant in case.variants:
                if len(variant.powers) < 3 or len(variant.vectors) < 3:
                    continue
                
                characters = [
                    Character(p, MatchupVector(v[0], v[1]))
                    for p, v in zip(variant.powers, variant.vectors)
                ]
                
                calc = aCalculator(characters[0], characters[1], characters[2])
                a = calc.calc()
                
                assert isinstance(a, MatchupVector), \
                    f"{case.name}/{variant.name}: 戻り値がMatchupVectorではありません"
                return  # 1つ検証できればOK
        
        pytest.skip("有効なテストケースが見つかりませんでした")
    
    def test_is_edge_property_exists(self):
        """
        is_edgeプロパティが存在する
        
        全ケースの最初の有効なvariantで検証。
        """
        all_cases = TheoryTestBuilder.get_all_cases()
        
        for case in all_cases:
            for variant in case.variants:
                if len(variant.powers) < 3 or len(variant.vectors) < 3:
                    continue
                
                characters = [
                    Character(p, MatchupVector(v[0], v[1]))
                    for p, v in zip(variant.powers, variant.vectors)
                ]
                
                calc = aCalculator(characters[0], characters[1], characters[2])
                calc.calc()
                
                # is_edgeプロパティにアクセスできる
                _ = calc.is_edge
                return  # 1つ検証できればOK
        
        pytest.skip("有効なテストケースが見つかりませんでした")
