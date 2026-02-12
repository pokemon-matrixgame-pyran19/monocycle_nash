from .base import EquilibriumSolver
from ..matrix.base import PayoffMatrix
from ..matrix.general import GeneralPayoffMatrix
from ..matrix.monocycle import MonocyclePayoffMatrix
from .nashpy_solver import NashpySolver
from .isopower_solver import IsopowerSolver


class SolverSelector:
    """
    利得行列の型に応じて最適なソルバーを選択
    - Strategy Patternによる自動選択
    """
    
    def __init__(self):
        self._isopower_solver = IsopowerSolver()
        self._nashpy_solver = NashpySolver()
    
    def select(self, matrix: PayoffMatrix) -> EquilibriumSolver:
        """
        行列型に応じたソルバーを選択
        - MonocyclePayoffMatrix → IsopowerSolver（高速）
        - GeneralPayoffMatrix → NashpySolver（汎用）
        """
        if isinstance(matrix, MonocyclePayoffMatrix):
            return self._isopower_solver
        return self._nashpy_solver
    
    def solve(self, matrix: PayoffMatrix):
        """適切なソルバーで均衡解を計算"""
        solver = self.select(matrix)
        return solver.solve(matrix)
