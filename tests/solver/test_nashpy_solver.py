import pytest
import numpy as np
from monocycle_nash.solver.nashpy_solver import NashpySolver
from monocycle_nash.matrix.general import GeneralPayoffMatrix
from monocycle_nash.matrix.monocycle import MonocyclePayoffMatrix
from monocycle_nash.character.domain import Character, MatchupVector


class TestNashpySolver:
    """NashpySolverクラスのテスト"""
    
    def test_can_solve_general_matrix(self):
        """GeneralPayoffMatrixは解ける"""
        solver = NashpySolver()
        matrix = GeneralPayoffMatrix(np.array([[0, 1], [-1, 0]], dtype=float))
        
        assert solver.can_solve(matrix) is True
    
    def test_can_solve_monocycle_matrix(self):
        """MonocyclePayoffMatrixも解ける"""
        solver = NashpySolver()
        characters = [
            Character(1.0, MatchupVector(1, 0)),
            Character(0.5, MatchupVector(0, 1)),
        ]
        matrix = MonocyclePayoffMatrix(characters)
        
        assert solver.can_solve(matrix) is True
    
    def test_solve_rock_paper_scissors(self):
        """じゃんけん行列の均衡解"""
        solver = NashpySolver()
        # 標準的なじゃんけん行列
        rps_matrix = np.array([
            [0, -1, 1],
            [1, 0, -1],
            [-1, 1, 0]
        ], dtype=float)
        matrix = GeneralPayoffMatrix(rps_matrix, ["Rock", "Paper", "Scissors"])
        
        result = solver.solve(matrix)
        
        # じゃんけんの均衡は各1/3
        assert result.validate()
        assert result.get_probability("Rock") == pytest.approx(1/3, abs=1e-6)
        assert result.get_probability("Paper") == pytest.approx(1/3, abs=1e-6)
        assert result.get_probability("Scissors") == pytest.approx(1/3, abs=1e-6)
    
    def test_solve_matching_pennies(self):
        """マッチングペニーの均衡解"""
        solver = NashpySolver()
        # マッチングペニー
        mp_matrix = np.array([
            [1, -1],
            [-1, 1]
        ], dtype=float)
        matrix = GeneralPayoffMatrix(mp_matrix, ["Heads", "Tails"])
        
        result = solver.solve(matrix)
        
        # 均衡は各1/2
        assert result.validate()
        assert result.get_probability("Heads") == pytest.approx(0.5, abs=1e-6)
        assert result.get_probability("Tails") == pytest.approx(0.5, abs=1e-6)
    
    def test_solve_trivial_game(self):
        """自明なゲーム（純粋戦略均衡）"""
        solver = NashpySolver()
        # 明確な最適戦略がある行列
        trivial_matrix = np.array([
            [2, 0],
            [0, 1]
        ], dtype=float)
        matrix = GeneralPayoffMatrix(trivial_matrix, ["A", "B"])
        
        result = solver.solve(matrix)
        
        # 均衡解が得られる
        assert result.validate()
        # 確率が正しく計算されていること
        assert 0 <= result.get_probability("A") <= 1
        assert 0 <= result.get_probability("B") <= 1
    
    def test_result_has_labels(self):
        """結果にラベルが含まれる"""
        solver = NashpySolver()
        matrix = GeneralPayoffMatrix(
            np.array([[0, 1], [-1, 0]], dtype=float),
            ["Player1", "Player2"]
        )
        
        result = solver.solve(matrix)
        
        assert result.strategy_ids == ["Player1", "Player2"]
    
    def test_result_type(self):
        """結果がMixedStrategy型"""
        solver = NashpySolver()
        matrix = GeneralPayoffMatrix(np.array([[0, 1], [-1, 0]], dtype=float))
        
        result = solver.solve(matrix)
        
        from monocycle_nash.equilibrium.domain import MixedStrategy
        assert isinstance(result, MixedStrategy)
    
    def test_result_probabilities_sum_to_one(self):
        """結果の確率合計が1"""
        solver = NashpySolver()
        matrix = GeneralPayoffMatrix(np.array([
            [0, 1, -1],
            [-1, 0, 1],
            [1, -1, 0]
        ], dtype=float))
        
        result = solver.solve(matrix)
        
        assert result.validate()
        assert abs(np.sum(result.probabilities) - 1.0) < 1e-6
