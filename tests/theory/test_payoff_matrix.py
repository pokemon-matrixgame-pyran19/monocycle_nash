"""
理論予測テスト - 利得行列計算の検証

TheoryTestBuilderの理論値を使用して、利得行列計算が正しいか検証。
新しいテストケースをTheoryTestBuilderに追加するだけで、
自動的にこのテストで検証される。
"""

import numpy as np
import pytest

from monocycle_nash.matrix.monocycle import MonocyclePayoffMatrix
from monocycle_nash.character.domain import Character, MatchupVector

from tests.theory import TheoryTestBuilder, TestVariant


class TestPayoffMatrixCalculation:
    """利得行列計算の理論予測テスト"""
    
    def _build_characters(self, variant: TestVariant) -> list[Character]:
        """variantからCharacterリストを生成"""
        return [
            Character(p, MatchupVector(list(v)))
            for p, v in zip(variant.powers, variant.vectors)
        ]
    
    def test_all_cases_all_variants_matrix_calculation(self):
        """
        全テストケースの全variantで利得行列計算を検証
        
        TheoryTestBuilder.get_all_cases()の各ケースについて、
        全variantでpowers + vectors → matrix の計算が正しいか検証。
        同じ利得行列になることを確認。
        """
        for case in TheoryTestBuilder.get_all_cases():
            for variant in case.variants:
                # 入力: powers, vectors
                characters = self._build_characters(variant)
                payoff = MonocyclePayoffMatrix(characters)
                
                # 出力: matrix が理論値と一致することを検証
                np.testing.assert_array_almost_equal(
                    payoff.matrix,
                    case.matrix,
                    decimal=10,
                    err_msg=f"{case.name}/{variant.name}: 利得行列が理論値と一致しない"
                )
    
    def test_all_cases_all_variants_diagonal_zero(self):
        """
        全variantで対角成分が0であることを検証
        
        理論的に A[i,i] = pi - pi + vi×vi = 0 となるはず。
        """
        for case in TheoryTestBuilder.get_all_cases():
            for variant in case.variants:
                characters = self._build_characters(variant)
                payoff = MonocyclePayoffMatrix(characters)
                
                for i in range(payoff.size):
                    assert payoff.matrix[i, i] == pytest.approx(0.0, abs=1e-10), \
                        f"{case.name}/{variant.name}: 対角成分[{i},{i}]が0ではない"
    
    def test_all_cases_all_variants_antisymmetric(self):
        """
        全variantで反対称性を検証
        
        理論的に A[i,j] = -A[j,i] となるはず。
        """
        for case in TheoryTestBuilder.get_all_cases():
            for variant in case.variants:
                characters = self._build_characters(variant)
                payoff = MonocyclePayoffMatrix(characters)
                
                matrix = payoff.matrix
                n = payoff.size
                for i in range(n):
                    for j in range(n):
                        if i != j:
                            assert matrix[i, j] == pytest.approx(-matrix[j, i], abs=1e-10), \
                                f"{case.name}/{variant.name}: 反対称性が成り立たない [{i},{j}]"
    
    def test_variants_produce_same_matrix(self):
        """
        同じケースの全variantが同一の利得行列を生成することを検証
        
        このテストは複数variantの存在意義を確認する核心的なテスト。
        """
        for case in TheoryTestBuilder.get_all_cases():
            if len(case.variants) <= 1:
                continue  # variantが1つだけならスキップ
            
            # 最初のvariantの行列を基準とする
            first_characters = self._build_characters(case.variants[0])
            first_matrix = MonocyclePayoffMatrix(first_characters).matrix
            
            # 残りのvariantも同じ行列になることを検証
            for variant in case.variants[1:]:
                characters = self._build_characters(variant)
                matrix = MonocyclePayoffMatrix(characters).matrix
                
                np.testing.assert_array_almost_equal(
                    matrix,
                    first_matrix,
                    decimal=10,
                    err_msg=f"{case.name}: {case.variants[0].name}と{variant.name}で行列が一致しない"
                )
