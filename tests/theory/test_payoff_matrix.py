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

from tests.theory import TheoryTestBuilder


class TestPayoffMatrixCalculation:
    """利得行列計算の理論予測テスト"""
    
    def _build_characters(self, powers, vectors) -> list[Character]:
        """パワーとベクトルからCharacterリストを生成"""
        return [
            Character(p, MatchupVector(v))
            for p, v in zip(powers, vectors)
        ]
    
    def test_all_cases_matrix_calculation(self):
        """
        全テストケースで利得行列計算を検証
        
        TheoryTestBuilder.get_all_cases()の各ケースについて、
        powers + vectors → matrix の計算が正しいか検証。
        """
        for case in TheoryTestBuilder.get_all_cases():
            # 入力: powers, vectors
            characters = self._build_characters(case.powers, case.vectors)
            payoff = MonocyclePayoffMatrix(characters)
            
            # 出力: matrix が理論値と一致することを検証
            np.testing.assert_array_almost_equal(
                payoff.matrix,
                case.matrix,
                decimal=10,
                err_msg=f"{case.name}: 利得行列が理論値と一致しない"
            )
    
    def test_all_cases_diagonal_zero(self):
        """
        対角成分が0であることを検証
        
        理論的に A[i,i] = pi - pi + vi×vi = 0 となるはず。
        """
        for case in TheoryTestBuilder.get_all_cases():
            characters = self._build_characters(case.powers, case.vectors)
            payoff = MonocyclePayoffMatrix(characters)
            
            for i in range(payoff.size):
                assert payoff.matrix[i, i] == pytest.approx(0.0, abs=1e-10), \
                    f"{case.name}: 対角成分[{i},{i}]が0ではない"
    
    def test_all_cases_antisymmetric(self):
        """
        反対称性を検証
        
        理論的に A[i,j] = -A[j,i] となるはず。
        """
        for case in TheoryTestBuilder.get_all_cases():
            characters = self._build_characters(case.powers, case.vectors)
            payoff = MonocyclePayoffMatrix(characters)
            
            matrix = payoff.matrix
            n = payoff.size
            for i in range(n):
                for j in range(n):
                    if i != j:
                        assert matrix[i, j] == pytest.approx(-matrix[j, i], abs=1e-10), \
                            f"{case.name}: 反対称性が成り立たない [{i},{j}]"
