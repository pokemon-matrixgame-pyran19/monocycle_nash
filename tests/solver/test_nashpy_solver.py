import pytest
import numpy as np
from matrix.general import GeneralPayoffMatrix
from solver.nashpy_solver import NashpySolver
from equilibrium.domain import MixedStrategy

def test_nashpy_solver_rps():
    # じゃんけん (Rock-Paper-Scissors)
    # 0: Rock, 1: Paper, 2: Scissors
    matrix = np.array([
        [0, -1, 1],
        [1, 0, -1],
        [-1, 1, 0]
    ])
    labels = ["Rock", "Paper", "Scissors"]
    payoff_matrix = GeneralPayoffMatrix(matrix, labels)

    solver = NashpySolver()
    result = solver.solve(payoff_matrix)

    assert isinstance(result, MixedStrategy)
    assert result.strategy_ids == labels
    # 均衡解は [1/3, 1/3, 1/3]
    expected = np.array([1/3, 1/3, 1/3])
    np.testing.assert_allclose(result.probabilities, expected, atol=1e-7)
    assert result.validate()

def test_nashpy_solver_dominant_strategy():
    # 優越戦略がある場合
    matrix = np.array([
        [2, 3],
        [1, 2]
    ])
    payoff_matrix = GeneralPayoffMatrix(matrix)

    solver = NashpySolver()
    result = solver.solve(payoff_matrix)

    assert isinstance(result, MixedStrategy)
    # 行1が常に有利なので [1, 0]
    expected = np.array([1.0, 0.0])
    np.testing.assert_allclose(result.probabilities, expected, atol=1e-7)

def test_nashpy_solver_asymmetric():
    # 非対称な行列
    matrix = np.array([
        [1, -2],
        [-3, 4]
    ])
    # このゲームを零和ゲームとして解く
    # Row player's payoff: A
    # Col player's payoff: -A
    # Equilibrium row strategy p:
    # 1*p - 3*(1-p) = -2*p + 4*(1-p)
    # p - 3 + 3p = -2p + 4 - 4p
    # 4p - 3 = -6p + 4
    # 10p = 7 => p = 0.7
    payoff_matrix = GeneralPayoffMatrix(matrix)

    solver = NashpySolver()
    result = solver.solve(payoff_matrix)

    assert isinstance(result, MixedStrategy)
    expected = np.array([0.7, 0.3])
    np.testing.assert_allclose(result.probabilities, expected, atol=1e-7)

def test_nashpy_solver_can_solve():
    solver = NashpySolver()
    matrix = GeneralPayoffMatrix(np.zeros((2, 2)))
    assert solver.can_solve(matrix) is True

def test_mixed_strategy_get_probability():
    probs = np.array([0.1, 0.9])
    ids = ["a", "b"]
    ms = MixedStrategy(probs, ids)
    assert ms.get_probability("a") == 0.1
    assert ms.get_probability("b") == 0.9
    assert ms.get_probability("c") == 0.0

def test_mixed_strategy_get_support():
    probs = np.array([0.5, 0.0, 0.5])
    ids = ["a", "b", "c"]
    ms = MixedStrategy(probs, ids)
    assert ms.get_support() == ["a", "c"]

def test_nashpy_solver_empty():
    matrix = np.zeros((0, 0))
    payoff_matrix = GeneralPayoffMatrix(matrix)

    solver = NashpySolver()
    result = solver.solve(payoff_matrix)

    assert result is None
