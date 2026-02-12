"""
理論予測テスト - ナッシュ均衡解の検証

TheoryTestBuilderの理論値を使用して、均衡解計算が正しいか検証。
新しいテストケースをTheoryTestBuilderに追加するだけで、
自動的にこのテストで検証される。
"""

import numpy as np
import pytest

from monocycle_nash.matrix.monocycle import MonocyclePayoffMatrix
from monocycle_nash.character.domain import Character, MatchupVector

from tests.theory import TheoryTestBuilder, TestVariant


class TestEquilibriumCalculation:
    """ナッシュ均衡解計算の理論予測テスト"""
    
    def _build_characters(self, variant: TestVariant) -> list[Character]:
        """variantからCharacterリストを生成"""
        return [
            Character(p, MatchupVector(list(v)))
            for p, v in zip(variant.powers, variant.vectors)
        ]
    
    def test_all_cases_all_variants_equilibrium(self):
        """
        全テストケースの全variantで均衡解を検証
        
        TheoryTestBuilder.get_all_cases()の各ケースについて、
        全variantで均衡解が同じ理論値になることを検証。
        equilibriumが定義されているケースのみ検証。
        """
        for case in TheoryTestBuilder.get_all_cases():
            if case.equilibrium is None:
                continue
            
            for variant in case.variants:
                # 入力: matrix（理論値を直接使用）
                characters = self._build_characters(variant)
                payoff = MonocyclePayoffMatrix(characters)
                
                # 均衡解を計算
                equilibrium = payoff.solve_equilibrium()
                
                # 出力: equilibrium が理論値と一致することを検証
                np.testing.assert_array_almost_equal(
                    equilibrium.probabilities,
                    case.equilibrium,
                    decimal=6,
                    err_msg=f"{case.name}/{variant.name}: 均衡解が理論値と一致しない"
                )
    
    def test_all_cases_all_variants_equilibrium_validity(self):
        """
        全variantで均衡解の妥当性を検証
        
        計算された均衡解が確率分布として妥当か検証。
        """
        for case in TheoryTestBuilder.get_all_cases():
            for variant in case.variants:
                characters = self._build_characters(variant)
                payoff = MonocyclePayoffMatrix(characters)
                
                equilibrium = payoff.solve_equilibrium()
                
                # 確率の合計が1に近いことを検証
                prob_sum = np.sum(equilibrium.probabilities)
                assert prob_sum == pytest.approx(1.0, abs=1e-6), \
                    f"{case.name}/{variant.name}: 確率の合計が1ではない ({prob_sum})"
                
                # 全ての確率が非負であることを検証
                assert np.all(equilibrium.probabilities >= -1e-10), \
                    f"{case.name}/{variant.name}: 負の確率が含まれている"
    
    def test_all_cases_all_variants_equilibrium_support(self):
        """
        全variantで均衡解のサポートを検証
        
        理論的に、単相性モデルでは最大3つの戦略がサポートを持つはず。
        """
        for case in TheoryTestBuilder.get_all_cases():
            for variant in case.variants:
                characters = self._build_characters(variant)
                payoff = MonocyclePayoffMatrix(characters)
                
                equilibrium = payoff.solve_equilibrium()
                support = equilibrium.get_support(threshold=1e-6)
                
                # サポートのサイズが3以下であることを検証
                assert len(support) <= 3, \
                    f"{case.name}/{variant.name}: サポートのサイズが3を超えている ({len(support)})"
    
    def test_variants_produce_same_equilibrium(self):
        """
        同じケースの全variantが同一の均衡解を生成することを検証
        
        利得行列が同じなら均衡解も同じはず。
        """
        for case in TheoryTestBuilder.get_all_cases():
            if case.equilibrium is None or len(case.variants) <= 1:
                continue
            
            # 最初のvariantの均衡解を基準とする
            first_characters = self._build_characters(case.variants[0])
            first_equilibrium = MonocyclePayoffMatrix(first_characters).solve_equilibrium()
            
            # 残りのvariantも同じ均衡解になることを検証
            for variant in case.variants[1:]:
                characters = self._build_characters(variant)
                equilibrium = MonocyclePayoffMatrix(characters).solve_equilibrium()
                
                np.testing.assert_array_almost_equal(
                    equilibrium.probabilities,
                    first_equilibrium.probabilities,
                    decimal=6,
                    err_msg=f"{case.name}: {case.variants[0].name}と{variant.name}で均衡解が一致しない"
                )
