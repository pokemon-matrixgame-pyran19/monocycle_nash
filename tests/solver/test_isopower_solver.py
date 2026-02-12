import pytest
import numpy as np
from monocycle_nash.solver.isopower_solver import IsopowerSolver
from monocycle_nash.matrix.monocycle import MonocyclePayoffMatrix
from monocycle_nash.matrix.general import GeneralPayoffMatrix
from monocycle_nash.character.domain import Character, MatchupVector


ROOT3 = 1.7320508075688772


class TestIsopowerSolver:
    """IsopowerSolverクラスのテスト"""
    
    def test_can_solve_monocycle_matrix(self):
        """MonocyclePayoffMatrixは解ける"""
        solver = IsopowerSolver()
        characters = [
            Character(1.0, MatchupVector(1, 0)),
            Character(0.5, MatchupVector(0, 1)),
        ]
        matrix = MonocyclePayoffMatrix(characters)
        
        assert solver.can_solve(matrix) is True
    
    def test_cannot_solve_general_matrix(self):
        """GeneralPayoffMatrixは解けない"""
        solver = IsopowerSolver()
        matrix = GeneralPayoffMatrix(np.array([[0, 1], [-1, 0]], dtype=float))
        
        assert solver.can_solve(matrix) is False
    
    def test_solve_simple_monocycle(self):
        """シンプルな単相性モデルの均衡解"""
        solver = IsopowerSolver()
        # 正三角形のキャラクター
        characters = [
            Character(1.0, MatchupVector(2, 0)),
            Character(1.0, MatchupVector(-1, ROOT3)),
            Character(1.0, MatchupVector(-1, -ROOT3)),
        ]
        matrix = MonocyclePayoffMatrix(characters)
        
        result = solver.solve(matrix)
        
        # 均衡解が得られる
        assert result.validate()
    
    def test_solve_with_power_difference(self):
        """パワー差がある場合の均衡解"""
        solver = IsopowerSolver()
        characters = [
            Character(2.0, MatchupVector(1, 0)),
            Character(1.0, MatchupVector(-0.5, ROOT3/2)),
            Character(0.0, MatchupVector(-0.5, -ROOT3/2)),
        ]
        matrix = MonocyclePayoffMatrix(characters, ["Strong", "Medium", "Weak"])
        
        result = solver.solve(matrix)
        
        # 均衡解が得られる
        assert result.validate()
        # サポートが3つ以内
        support = result.get_support()
        assert len(support) <= 3
    
    def test_fallback_when_no_triangle(self):
        """有効な三角形がない場合はフォールバック"""
        solver = IsopowerSolver()
        # 2キャラクターだけ（三角形が形成できない）
        characters = [
            Character(1.0, MatchupVector(1, 0)),
            Character(0.5, MatchupVector(0, 1)),
        ]
        matrix = MonocyclePayoffMatrix(characters)
        
        result = solver.solve(matrix)
        
        # フォールバック（均等分布）が返される
        assert result.validate()
    
    def test_result_has_correct_labels(self):
        """結果に正しいラベルが含まれる"""
        solver = IsopowerSolver()
        characters = [
            Character(1.0, MatchupVector(2, 0)),
            Character(1.0, MatchupVector(-1, ROOT3)),
            Character(1.0, MatchupVector(-1, -ROOT3)),
        ]
        labels = ["A", "B", "C"]
        matrix = MonocyclePayoffMatrix(characters, labels)
        
        result = solver.solve(matrix)
        
        assert result.strategy_ids == labels
    
    def test_result_type(self):
        """結果がMixedStrategy型"""
        solver = IsopowerSolver()
        characters = [
            Character(1.0, MatchupVector(2, 0)),
            Character(1.0, MatchupVector(-1, ROOT3)),
            Character(1.0, MatchupVector(-1, -ROOT3)),
        ]
        matrix = MonocyclePayoffMatrix(characters)
        
        result = solver.solve(matrix)
        
        from monocycle_nash.equilibrium.domain import MixedStrategy
        assert isinstance(result, MixedStrategy)
    
    def test_result_probabilities_sum_to_one(self):
        """結果の確率合計が1"""
        solver = IsopowerSolver()
        characters = [
            Character(1.0, MatchupVector(2, 0)),
            Character(1.0, MatchupVector(-1, ROOT3)),
            Character(1.0, MatchupVector(-1, -ROOT3)),
        ]
        matrix = MonocyclePayoffMatrix(characters)
        
        result = solver.solve(matrix)
        
        assert result.validate()
        assert abs(np.sum(result.probabilities) - 1.0) < 1e-6
    
    def test_support_size(self):
        """サポートサイズが3以下"""
        solver = IsopowerSolver()
        characters = [
            Character(1.0, MatchupVector(2, 0)),
            Character(1.0, MatchupVector(-1, ROOT3)),
            Character(1.0, MatchupVector(-1, -ROOT3)),
            Character(0.5, MatchupVector(0, 0)),
        ]
        matrix = MonocyclePayoffMatrix(characters)
        
        result = solver.solve(matrix)
        
        # 最適三角形の3つ以内
        support = result.get_support()
        assert len(support) <= 3
