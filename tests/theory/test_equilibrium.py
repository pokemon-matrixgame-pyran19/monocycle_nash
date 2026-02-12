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

from tests.theory import TheoryTestBuilder


class TestEquilibriumCalculation:
    """ナッシュ均衡解計算の理論予測テスト"""
    
    def _build_characters(self, powers, vectors) -> list[Character]:
        """パワーとベクトルからCharacterリストを生成"""
        return [
            Character(p, MatchupVector(v))
            for p, v in zip(powers, vectors)
        ]
    
    def test_all_cases_equilibrium(self):
        """
        全テストケースで均衡解を検証
        
        TheoryTestBuilder.get_all_cases()の各ケースについて、
        matrix → equilibrium の計算が正しいか検証。
        equilibriumが定義されているケースのみ検証。
        """
        for case in TheoryTestBuilder.get_all_cases():
            if case.equilibrium is None:
                continue
            
            # 入力: matrix（理論値を直接使用）
            characters = self._build_characters(case.powers, case.vectors)
            payoff = MonocyclePayoffMatrix(characters)
            
            # 均衡解を計算
            equilibrium = payoff.solve_equilibrium()
            
            # 出力: equilibrium が理論値と一致することを検証
            np.testing.assert_array_almost_equal(
                equilibrium.probabilities,
                case.equilibrium,
                decimal=6,
                err_msg=f"{case.name}: 均衡解が理論値と一致しない"
            )
    
    def test_all_cases_equilibrium_validity(self):
        """
        均衡解の妥当性を検証
        
        計算された均衡解が確率分布として妥当か検証。
        """
        for case in TheoryTestBuilder.get_all_cases():
            characters = self._build_characters(case.powers, case.vectors)
            payoff = MonocyclePayoffMatrix(characters)
            
            equilibrium = payoff.solve_equilibrium()
            
            # 確率の合計が1に近いことを検証
            prob_sum = np.sum(equilibrium.probabilities)
            assert prob_sum == pytest.approx(1.0, abs=1e-6), \
                f"{case.name}: 確率の合計が1ではない ({prob_sum})"
            
            # 全ての確率が非負であることを検証
            assert np.all(equilibrium.probabilities >= -1e-10), \
                f"{case.name}: 負の確率が含まれている"
    
    def test_all_cases_equilibrium_support(self):
        """
        均衡解のサポート（正の確率を持つ戦略）を検証
        
        理論的に、単相性モデルでは最大3つの戦略がサポートを持つはず。
        """
        for case in TheoryTestBuilder.get_all_cases():
            characters = self._build_characters(case.powers, case.vectors)
            payoff = MonocyclePayoffMatrix(characters)
            
            equilibrium = payoff.solve_equilibrium()
            support = equilibrium.get_support(threshold=1e-6)
            
            # サポートのサイズが3以下であることを検証
            assert len(support) <= 3, \
                f"{case.name}: サポートのサイズが3を超えている ({len(support)})"
