import pytest
import numpy as np
from monocycle_nash.solver.selector import SolverSelector
from monocycle_nash.solver.nashpy_solver import NashpySolver
from monocycle_nash.solver.isopower_solver import IsopowerSolver
from monocycle_nash.matrix.general import GeneralPayoffMatrix
from monocycle_nash.matrix.monocycle import MonocyclePayoffMatrix
from monocycle_nash.character.domain import Character, MatchupVector


ROOT3 = 1.7320508075688772


class TestSolverSelector:
    """SolverSelectorクラスのテスト"""
    
    def test_select_isopower_for_monocycle(self):
        """MonocyclePayoffMatrixにはIsopowerSolverを選択"""
        selector = SolverSelector()
        characters = [
            Character(1.0, MatchupVector(1, 0)),
            Character(0.5, MatchupVector(0, 1)),
        ]
        matrix = MonocyclePayoffMatrix(characters)
        
        solver = selector.select(matrix)
        
        assert isinstance(solver, IsopowerSolver)
    
    def test_select_nashpy_for_general(self):
        """GeneralPayoffMatrixにはNashpySolverを選択"""
        selector = SolverSelector()
        matrix = GeneralPayoffMatrix(np.array([[0, 1], [-1, 0]], dtype=float))
        
        solver = selector.select(matrix)
        
        assert isinstance(solver, NashpySolver)
    
    def test_solve_monocycle(self):
        """MonocyclePayoffMatrixを解く"""
        selector = SolverSelector()
        characters = [
            Character(1.0, MatchupVector(2, 0)),
            Character(1.0, MatchupVector(-1, ROOT3)),
            Character(1.0, MatchupVector(-1, -ROOT3)),
        ]
        matrix = MonocyclePayoffMatrix(characters)
        
        result = selector.solve(matrix)
        
        # 均衡解が得られる
        assert result.validate()
    
    def test_solve_general(self):
        """GeneralPayoffMatrixを解く"""
        selector = SolverSelector()
        matrix = GeneralPayoffMatrix(
            np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]], dtype=float),
            ["A", "B", "C"]
        )
        
        result = selector.solve(matrix)
        
        # 均衡解が得られる
        assert result.validate()
        # じゃんけんなので各1/3
        assert result.get_probability("A") == pytest.approx(1/3, abs=1e-5)
        assert result.get_probability("B") == pytest.approx(1/3, abs=1e-5)
        assert result.get_probability("C") == pytest.approx(1/3, abs=1e-5)
    
    def test_solve_result_type(self):
        """solveメソッドの結果型"""
        selector = SolverSelector()
        
        # Monocycle
        characters = [
            Character(1.0, MatchupVector(1, 0)),
            Character(0.5, MatchupVector(0, 1)),
        ]
        mono_matrix = MonocyclePayoffMatrix(characters)
        mono_result = selector.solve(mono_matrix)
        
        # General
        gen_matrix = GeneralPayoffMatrix(np.array([[0, 1], [-1, 0]], dtype=float))
        gen_result = selector.solve(gen_matrix)
        
        from monocycle_nash.equilibrium.domain import MixedStrategy
        assert isinstance(mono_result, MixedStrategy)
        assert isinstance(gen_result, MixedStrategy)
    
    def test_selector_returns_solver(self):
        """selectメソッドがソルバーを返す"""
        selector = SolverSelector()
        
        from monocycle_nash.solver.base import EquilibriumSolver
        
        characters = [Character(1.0, MatchupVector(1, 0))]
        mono_matrix = MonocyclePayoffMatrix(characters)
        gen_matrix = GeneralPayoffMatrix(np.array([[0]], dtype=float))
        
        mono_solver = selector.select(mono_matrix)
        gen_solver = selector.select(gen_matrix)
        
        assert isinstance(mono_solver, EquilibriumSolver)
        assert isinstance(gen_solver, EquilibriumSolver)
