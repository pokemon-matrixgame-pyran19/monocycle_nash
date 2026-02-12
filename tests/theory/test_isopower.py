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

from tests.theory import TheoryTestBuilder


class TestIsopowerCoordinate:
    """等パワー座標計算の理論予測テスト"""
    
    def _build_characters(self, powers, vectors) -> list[Character]:
        """パワーとベクトルからCharacterリストを生成"""
        return [
            Character(p, MatchupVector(v))
            for p, v in zip(powers, vectors)
        ]
    
    def test_all_cases_isopower_coordinate(self):
        """
        全テストケースで等パワー座標を検証
        
        TheoryTestBuilder.get_all_cases()の各ケースについて、
        powers[:3], vectors[:3] → isopower_a の計算が正しいか検証。
        isopower_aが定義されているケースのみ検証。
        """
        for case in TheoryTestBuilder.get_all_cases():
            if case.isopower_a is None:
                continue
            
            # 入力: powers, vectors（先頭3点）
            characters = self._build_characters(
                case.powers[:3],
                case.vectors[:3]
            )
            payoff = MonocyclePayoffMatrix(characters)
            
            # 等パワー座標を取得（MonocyclePayoffMatrix経由で）
            # 注: 現在の実装では直接取得するメソッドがないため、
            # 原点移動後のパワーが等しくなることを間接的に検証
            
            # 理論値の等パワー座標で原点移動
            a = MatchupVector(case.isopower_a[0], case.isopower_a[1])
            shifted = payoff.shift_origin(a)
            
            # 移動後のパワーが等しくなることを検証
            shifted_powers = shifted.get_power_vector()
            
            # 全てのパワーが等しいことを検証
            for i in range(1, len(shifted_powers)):
                assert shifted_powers[i] == pytest.approx(shifted_powers[0], abs=1e-6), \
                    f"{case.name}: 原点移動後のパワーが等しくない"
    
    def test_all_cases_isopower_preserves_matrix(self):
        """
        等パワー座標による原点移動が利得行列を保存することを検証
        
        理論的に、原点移動しても利得行列は変わらないはず。
        """
        for case in TheoryTestBuilder.get_all_cases():
            if case.isopower_a is None:
                continue
            
            characters = self._build_characters(
                case.powers[:3],
                case.vectors[:3]
            )
            payoff = MonocyclePayoffMatrix(characters)
            
            # 原点移動前の行列を保存
            original_matrix = payoff.matrix.copy()
            
            # 等パワー座標で原点移動
            a = MatchupVector(case.isopower_a[0], case.isopower_a[1])
            shifted = payoff.shift_origin(a)
            
            # 行列が保存されることを検証
            np.testing.assert_array_almost_equal(
                shifted.matrix,
                original_matrix,
                decimal=10,
                err_msg=f"{case.name}: 原点移動後の行列が変化している"
            )
    
    def test_janken_isopower_is_origin(self):
        """
        じゃんけんの等パワー座標が原点であることを検証
        
        正三角形かつ等パワーの場合、等パワー座標は原点(0,0)となる。
        """
        case = TheoryTestBuilder.janken()
        
        assert case.isopower_a is not None
        assert case.isopower_a[0] == pytest.approx(0.0, abs=1e-10)
        assert case.isopower_a[1] == pytest.approx(0.0, abs=1e-10)
