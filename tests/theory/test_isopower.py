"""
理論予測テスト - 等パワー座標の検証

TheoryTestBuilderの理論値を使用して、等パワー座標計算が正しいか検証。
新しいテストケースをTheoryTestBuilderに追加するだけで、
自動的にこのテストで検証される。
"""

import numpy as np
import pytest

from monocycle_nash.matrix.monocycle import MonocyclePayoffMatrix
from monocycle_nash.character.domain import Character, MatchupVector

from tests.theory import TheoryTestBuilder, TestVariant


class TestIsopowerCoordinate:
    """等パワー座標計算の理論予測テスト"""
    
    def _build_characters(self, variant: TestVariant) -> list[Character]:
        """variantからCharacterリストを生成"""
        return [
            Character(p, MatchupVector(list(v)))
            for p, v in zip(variant.powers, variant.vectors)
        ]
    
    def test_all_cases_all_variants_isopower_coordinate(self):
        """
        全テストケースの全variantで等パワー座標を検証
        
        TheoryTestBuilder.get_all_cases()の各ケースについて、
        各variantのisopower_aで原点移動したときに等パワーになることを検証。
        """
        for case in TheoryTestBuilder.get_all_cases():
            for variant in case.variants:
                if variant.isopower_a is None:
                    continue
                
                # 入力: powers, vectors（先頭3点）
                characters = self._build_characters(variant)
                payoff = MonocyclePayoffMatrix(characters)
                
                # 等パワー座標で原点移動
                a = MatchupVector(variant.isopower_a[0], variant.isopower_a[1])
                shifted = payoff.shift_origin(a)
                
                # 移動後のパワーが等しくなることを検証
                shifted_powers = shifted.get_power_vector()
                
                # 全てのパワーが等しいことを検証
                for i in range(1, len(shifted_powers)):
                    assert shifted_powers[i] == pytest.approx(shifted_powers[0], abs=1e-6), \
                        f"{case.name}/{variant.name}: 原点移動後のパワーが等しくない"
    
    def test_all_cases_all_variants_isopower_preserves_matrix(self):
        """
        全variantで等パワー座標による原点移動が利得行列を保存することを検証
        
        理論的に、原点移動しても利得行列は変わらないはず。
        """
        for case in TheoryTestBuilder.get_all_cases():
            for variant in case.variants:
                if variant.isopower_a is None:
                    continue
                
                characters = self._build_characters(variant)
                payoff = MonocyclePayoffMatrix(characters)
                
                # 原点移動前の行列を保存
                original_matrix = payoff.matrix.copy()
                
                # 等パワー座標で原点移動
                a = MatchupVector(variant.isopower_a[0], variant.isopower_a[1])
                shifted = payoff.shift_origin(a)
                
                # 行列が保存されることを検証
                np.testing.assert_array_almost_equal(
                    shifted.matrix,
                    original_matrix,
                    decimal=10,
                    err_msg=f"{case.name}/{variant.name}: 原点移動後の行列が変化している"
                )
    
    def test_variants_isopower_consistency(self):
        """
        全variantの等パワー化が同じパワー値を生成することを検証
        
        同じ利得行列なので、等パワー化後のパワー値も同じはず。
        """
        for case in TheoryTestBuilder.get_all_cases():
            if len(case.variants) <= 1:
                continue
            
            isopower_values = []
            
            for variant in case.variants:
                if variant.isopower_a is None:
                    continue
                
                characters = self._build_characters(variant)
                payoff = MonocyclePayoffMatrix(characters)
                
                # 等パワー座標で原点移動
                a = MatchupVector(variant.isopower_a[0], variant.isopower_a[1])
                shifted = payoff.shift_origin(a)
                
                # 等パワー化後のパワー値を記録
                shifted_powers = shifted.get_power_vector()
                isopower_values.append((variant.name, shifted_powers[0]))
            
            # 少なくとも2つのvariantが検証できた場合
            if len(isopower_values) >= 2:
                first_value = isopower_values[0][1]
                for name, value in isopower_values[1:]:
                    assert value == pytest.approx(first_value, abs=1e-6), \
                        f"{case.name}: {isopower_values[0][0]}と{name}で等パワー値が一致しない"


class TestVariantTransformations:
    """variant間の変換関係テスト"""
    
    def _build_characters(self, variant: TestVariant) -> list[Character]:
        """variantからCharacterリストを生成"""
        return [
            Character(p, MatchupVector(list(v)))
            for p, v in zip(variant.powers, variant.vectors)
        ]
    
    def test_declared_transformations(self):
        """
        定義された変換関係が正しいことを検証
        
        transformationsで定義された(from, to, shift)について、
        fromをshiftだけ平行移動するとtoになることを検証。
        """
        for case in TheoryTestBuilder.get_all_cases():
            if case.transformations is None:
                continue
            
            for from_idx, to_idx, shift_vector in case.transformations:
                from_variant = case.variants[from_idx]
                to_variant = case.variants[to_idx]
                
                # from_variantをshift_vectorだけ平行移動
                from_characters = self._build_characters(from_variant)
                from_payoff = MonocyclePayoffMatrix(from_characters)
                
                shift = MatchupVector(shift_vector[0], shift_vector[1])
                transformed = from_payoff.shift_origin(shift)
                
                # to_variantと同じになることを検証
                to_characters = self._build_characters(to_variant)
                to_payoff = MonocyclePayoffMatrix(to_characters)
                
                # パワーベクトルの比較
                np.testing.assert_array_almost_equal(
                    transformed.get_power_vector(),
                    to_payoff.get_power_vector(),
                    decimal=10,
                    err_msg=f"{case.name}: {from_variant.name}を平行移動しても{to_variant.name}のパワーと一致しない"
                )
                
                # 相性ベクトルの比較
                np.testing.assert_array_almost_equal(
                    transformed.get_matchup_vectors(),
                    to_payoff.get_matchup_vectors(),
                    decimal=10,
                    err_msg=f"{case.name}: {from_variant.name}を平行移動しても{to_variant.name}のベクトルと一致しない"
                )
                
                # 利得行列も一致することを確認
                np.testing.assert_array_almost_equal(
                    transformed.matrix,
                    to_payoff.matrix,
                    decimal=10,
                    err_msg=f"{case.name}: {from_variant.name}を平行移動しても{to_variant.name}の行列と一致しない"
                )
    
    def test_janken_original_to_shifted(self):
        """
        じゃんけん: originalを平行移動するとshiftedになることを直接検証
        
        具体的な数値で変換関係を確認。
        """
        case = TheoryTestBuilder.janken()
        
        original = case.variants[0]  # original
        shifted = case.variants[1]   # shifted
        
        # originalを(1, 1)だけ平行移動
        original_characters = self._build_characters(original)
        original_payoff = MonocyclePayoffMatrix(original_characters)
        
        shift = MatchupVector(1.0, 1.0)
        transformed = original_payoff.shift_origin(shift)
        
        # shiftedと同じになることを検証
        shifted_characters = self._build_characters(shifted)
        shifted_payoff = MonocyclePayoffMatrix(shifted_characters)
        
        # パワーが[0, 0, 0]になることを検証
        transformed_powers = transformed.get_power_vector()
        np.testing.assert_array_almost_equal(
            transformed_powers,
            np.array([0.0, 0.0, 0.0]),
            decimal=10,
            err_msg="originalを(1,1)移動後のパワーが[0,0,0]ではない"
        )
        
        # ベクトルが一致することを検証
        np.testing.assert_array_almost_equal(
            transformed.get_matchup_vectors(),
            shifted_payoff.get_matchup_vectors(),
            decimal=10,
            err_msg="originalを(1,1)移動後のベクトルがshiftedと一致しない"
        )
    
    def test_janken_shifted_is_already_isopower(self):
        """
        じゃんけん: shifted variantが既に等パワーであることを検証
        
        shiftedのパワーが全て0であることを確認。
        """
        case = TheoryTestBuilder.janken()
        shifted = case.variants[1]  # shifted
        
        shifted_characters = self._build_characters(shifted)
        shifted_payoff = MonocyclePayoffMatrix(shifted_characters)
        
        shifted_powers = shifted_payoff.get_power_vector()
        
        # 全てのパワーが0に近いことを検証
        for i, p in enumerate(shifted_powers):
            assert p == pytest.approx(0.0, abs=1e-10), \
                f"shiftedのパワー[{i}]が0ではない ({p})"
        
        # isopower_aも(0, 0)であることを検証
        assert shifted.isopower_a is not None
        assert shifted.isopower_a[0] == pytest.approx(0.0, abs=1e-10)
        assert shifted.isopower_a[1] == pytest.approx(0.0, abs=1e-10)
