"""
両ソルバーの結果が同じになることを検証するテスト

NashpySolver（汎用）とIsopowerSolver（単相性モデル専用）で
同じ単相性モデルの利得行列を解いた場合、同じ均衡解が得られることを確認します。
"""

import pytest
import numpy as np
from monocycle_nash.solver.nashpy_solver import NashpySolver
from monocycle_nash.solver.isopower_solver import IsopowerSolver
from monocycle_nash.matrix.monocycle import MonocyclePayoffMatrix
from monocycle_nash.character.domain import Character, MatchupVector


ROOT3 = 1.7320508075688772


class TestSolverEquivalence:
    """
    NashpySolverとIsopowerSolverの結果が同じになることを検証
    """

    def test_simple_triangle(self):
        """正三角形配置のキャラクターで両ソルバーの結果が一致"""
        # 正三角形のキャラクター（同じパワー）
        characters = [
            Character(1.0, MatchupVector(2, 0)),
            Character(1.0, MatchupVector(-1, ROOT3)),
            Character(1.0, MatchupVector(-1, -ROOT3)),
        ]
        matrix = MonocyclePayoffMatrix(characters, ["A", "B", "C"])
        
        # 両ソルバーで解く
        nash_result = NashpySolver().solve(matrix)
        isopower_result = IsopowerSolver().solve(matrix)
        
        # 両方とも有効な均衡解
        assert nash_result.validate()
        assert isopower_result.validate()
        
        # サポートが同じ（使用する戦略が同じ）
        nash_support = set(nash_result.get_support())
        isopower_support = set(isopower_result.get_support())
        assert nash_support == isopower_support, \
            f"Support mismatch: {nash_support} vs {isopower_support}"
        
        # 確率分布がほぼ同じ（許容誤差内）
        for label in ["A", "B", "C"]:
            nash_prob = nash_result.get_probability(label)
            isopower_prob = isopower_result.get_probability(label)
            assert nash_prob == pytest.approx(isopower_prob, abs=1e-5), \
                f"Probability mismatch for {label}: {nash_prob} vs {isopower_prob}"

    def test_power_difference(self):
        """パワー差がある場合も両ソルバーが有効な均衡解を返す"""
        characters = [
            Character(2.0, MatchupVector(1, 0)),
            Character(1.0, MatchupVector(-0.5, ROOT3/2)),
            Character(0.0, MatchupVector(-0.5, -ROOT3/2)),
        ]
        matrix = MonocyclePayoffMatrix(characters, ["Strong", "Medium", "Weak"])
        
        # 両ソルバーで解く
        nash_result = NashpySolver().solve(matrix)
        isopower_result = IsopowerSolver().solve(matrix)
        
        # 両方とも有効な均衡解
        assert nash_result.validate()
        assert isopower_result.validate()
        
        # 両方ともサポートを持つ
        nash_support = nash_result.get_support()
        isopower_support = isopower_result.get_support()
        assert len(nash_support) >= 1
        assert len(isopower_support) >= 1
        # IsopowerSolverは最大3つまで
        assert len(isopower_support) <= 3

    def test_four_characters(self):
        """4キャラクターの場合も両ソルバーの結果が一致"""
        characters = [
            Character(1.0, MatchupVector(2, 0)),
            Character(1.0, MatchupVector(-1, ROOT3)),
            Character(1.0, MatchupVector(-1, -ROOT3)),
            Character(0.5, MatchupVector(0, 0)),
        ]
        matrix = MonocyclePayoffMatrix(characters, ["A", "B", "C", "D"])
        
        # 両ソルバーで解く
        nash_result = NashpySolver().solve(matrix)
        isopower_result = IsopowerSolver().solve(matrix)
        
        # 両方とも有効な均衡解
        assert nash_result.validate()
        assert isopower_result.validate()
        
        # サポートが同じ（最適な3キャラクターが選ばれる）
        nash_support = set(nash_result.get_support())
        isopower_support = set(isopower_result.get_support())
        assert nash_support == isopower_support, \
            f"Support mismatch: {nash_support} vs {isopower_support}"

    def test_collinear_characters(self):
        """3点が一直線上にある場合（特殊ケース）"""
        # 共線なキャラクター
        characters = [
            Character(1.0, MatchupVector(1, 0)),
            Character(0.5, MatchupVector(0, 0)),
            Character(0.0, MatchupVector(-1, 0)),
        ]
        matrix = MonocyclePayoffMatrix(characters, ["A", "B", "C"])
        
        # 両ソルバーで解く（この場合はフォールバック動作になる）
        nash_result = NashpySolver().solve(matrix)
        isopower_result = IsopowerSolver().solve(matrix)
        
        # 両方とも有効な均衡解
        assert nash_result.validate()
        assert isopower_result.validate()

    def test_probability_sum_is_one(self):
        """両ソルバーの結果の確率合計が1になる"""
        characters = [
            Character(1.0, MatchupVector(1, 0)),
            Character(0.5, MatchupVector(-0.5, ROOT3/2)),
            Character(0.0, MatchupVector(-0.5, -ROOT3/2)),
        ]
        matrix = MonocyclePayoffMatrix(characters)
        
        nash_result = NashpySolver().solve(matrix)
        isopower_result = IsopowerSolver().solve(matrix)
        
        # 確率合計が1
        assert abs(np.sum(nash_result.probabilities) - 1.0) < 1e-6
        assert abs(np.sum(isopower_result.probabilities) - 1.0) < 1e-6

    def test_both_return_mixed_strategy(self):
        """両ソルバーがMixedStrategy型を返す"""
        characters = [
            Character(1.0, MatchupVector(1, 0)),
            Character(0.5, MatchupVector(0, 1)),
            Character(0.0, MatchupVector(-0.5, -0.5)),
        ]
        matrix = MonocyclePayoffMatrix(characters)
        
        nash_result = NashpySolver().solve(matrix)
        isopower_result = IsopowerSolver().solve(matrix)
        
        from monocycle_nash.equilibrium.domain import MixedStrategy
        assert isinstance(nash_result, MixedStrategy)
        assert isinstance(isopower_result, MixedStrategy)

    def test_support_size_limit(self):
        """サポートサイズが3以下になる"""
        # 多くのキャラクター
        characters = [
            Character(1.0, MatchupVector(2, 0)),
            Character(1.0, MatchupVector(-1, ROOT3)),
            Character(1.0, MatchupVector(-1, -ROOT3)),
            Character(0.8, MatchupVector(0.5, 0.5)),
            Character(0.6, MatchupVector(-0.5, 0.5)),
        ]
        matrix = MonocyclePayoffMatrix(characters)
        
        nash_result = NashpySolver().solve(matrix)
        isopower_result = IsopowerSolver().solve(matrix)
        
        # サポートサイズが3以下
        assert len(nash_result.get_support()) <= 3
        assert len(isopower_result.get_support()) <= 3
