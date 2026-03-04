import numpy as np
import pytest

from monocycle_nash.character.domain import Character, MatchupVector
from monocycle_nash.matrix.approximation import (
    ApproximationQualityEvaluator,
    MaxElementDifferenceDistance,
    MonocycleToGeneralApproximation,
    DominantEigenpairMonocycleApproximation,
    EquilibriumPreservingResidualMonocycleApproximation,
    EquilibriumUStrategyDifferenceDistance,
)
from monocycle_nash.matrix.builder import PayoffMatrixBuilder
from monocycle_nash.matrix.general import GeneralPayoffMatrix

from monocycle_nash.equilibrium.domain import MixedStrategy


class _StubSolverSelector:
    def __init__(self, probabilities: np.ndarray):
        self._probabilities = probabilities

    def solve(self, matrix):
        return MixedStrategy(self._probabilities, matrix.col_strategies.ids)


@pytest.fixture
def sample_monocycle_matrix():
    characters = [
        Character(1.0, MatchupVector(1.0, 0.0), label="A"),
        Character(0.2, MatchupVector(0.0, 1.0), label="B"),
        Character(0.8, MatchupVector(-1.0, 0.0), label="C"),
    ]
    return PayoffMatrixBuilder.from_characters(characters)


def test_monocycle_to_general_approximation_returns_general_matrix(sample_monocycle_matrix):
    approx = MonocycleToGeneralApproximation()

    result = approx.approximate(sample_monocycle_matrix)

    assert isinstance(result, GeneralPayoffMatrix)
    assert np.array_equal(result.matrix, sample_monocycle_matrix.matrix)
    assert result.row_strategies is sample_monocycle_matrix.row_strategies




def test_max_element_difference_distance_rejects_shape_mismatch():
    distance = MaxElementDifferenceDistance()
    left = GeneralPayoffMatrix(np.array([[1.0, 2.0], [3.0, 4.0]]))
    right = GeneralPayoffMatrix(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))

    with pytest.raises(ValueError):
        distance.calculate(left, right)


def test_max_element_difference_distance_uses_sup_norm():
    distance = MaxElementDifferenceDistance()
    left = GeneralPayoffMatrix(np.array([[1.0, 4.0], [0.0, -1.0]]))
    right = GeneralPayoffMatrix(np.array([[2.0, 2.0], [3.0, -2.0]]))

    assert distance.calculate(left, right) == 3.0


def test_approximation_quality_evaluator_combines_approximation_and_distance(sample_monocycle_matrix):
    approximation = MonocycleToGeneralApproximation()
    distance = MaxElementDifferenceDistance()
    evaluator = ApproximationQualityEvaluator(approximation, distance)
    reference = GeneralPayoffMatrix(sample_monocycle_matrix.matrix + 0.5)

    quality = evaluator.evaluate(sample_monocycle_matrix, reference)

    assert quality == pytest.approx(0.5)


def test_dominant_eigenpair_monocycle_approximation_extracts_largest_pair_component():
    approximation = DominantEigenpairMonocycleApproximation()
    matrix = np.array(
        [
            [0.0, -5.0, 0.0, 0.0],
            [5.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, -2.0],
            [0.0, 0.0, 2.0, 0.0],
        ]
    )
    source = GeneralPayoffMatrix(matrix)

    result = approximation.approximate(source)

    expected = np.array(
        [
            [0.0, -5.0, 0.0, 0.0],
            [5.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )
    assert np.allclose(result.matrix, expected, atol=1e-7)


def test_dominant_eigenpair_monocycle_approximation_rejects_non_alternating_matrix():
    approximation = DominantEigenpairMonocycleApproximation()
    source = GeneralPayoffMatrix(np.array([[0.0, 1.0], [1.0, 0.0]]))

    with pytest.raises(ValueError):
        approximation.approximate(source)


def test_dominant_eigenpair_monocycle_approximation_quality_parameters_provides_histogram_bin():
    approximation = DominantEigenpairMonocycleApproximation()
    matrix = np.array(
        [
            [0.0, -5.0, 0.0, 0.0],
            [5.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, -2.0],
            [0.0, 0.0, 2.0, 0.0],
        ]
    )
    source = GeneralPayoffMatrix(matrix)

    parameters = approximation.quality_parameters(
        source,
        config={"dominant_eigen_ratio_bin_edges": [1.5, 2.0, 3.0]},
    )

    assert parameters["dominant_eigen_ratio"] == pytest.approx(2.5)
    assert parameters["dominant_eigen_ratio_bin"] == "[2.000,3.000)"


def test_equilibrium_u_strategy_difference_distance_uses_sup_norm_on_expected_payoff_gap():
    left = GeneralPayoffMatrix(np.array([[0.0, 2.0], [-2.0, 0.0]]))
    right = GeneralPayoffMatrix(np.array([[0.0, 1.0], [-1.0, 0.0]]))
    selector = _StubSolverSelector(np.array([0.25, 0.75]))
    distance = EquilibriumUStrategyDifferenceDistance(solver_selector=selector)

    # (left-right)@u = [0.75, -0.75], sup norm = 0.75
    assert distance.calculate(left, right) == pytest.approx(0.75)


def test_equilibrium_u_strategy_difference_distance_rejects_shape_mismatch():
    distance = EquilibriumUStrategyDifferenceDistance(solver_selector=_StubSolverSelector(np.array([0.5, 0.5])))
    left = GeneralPayoffMatrix(np.array([[1.0, 2.0], [3.0, 4.0]]))
    right = GeneralPayoffMatrix(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))

    with pytest.raises(ValueError):
        distance.calculate(left, right)


def test_equilibrium_preserving_residual_monocycle_approximation_matches_equilibrium_action():
    matrix = np.array(
        [
            [0.0, -5.0, 0.7, -0.1],
            [5.0, 0.0, 0.3, -0.4],
            [-0.7, -0.3, 0.0, -2.0],
            [0.1, 0.4, 2.0, 0.0],
        ]
    )
    source = GeneralPayoffMatrix(matrix)
    u = np.array([0.1, 0.2, 0.3, 0.4])

    approximation = EquilibriumPreservingResidualMonocycleApproximation(
        solver_selector=_StubSolverSelector(u)
    )
    result = approximation.approximate(source)

    assert np.allclose(result.matrix + result.matrix.T, 0.0, atol=1e-8)
    assert np.allclose(result.matrix @ u, source.matrix @ u, atol=1e-7)


def test_equilibrium_preserving_residual_monocycle_approximation_rejects_non_alternating_matrix():
    approximation = EquilibriumPreservingResidualMonocycleApproximation(
        solver_selector=_StubSolverSelector(np.array([0.5, 0.5]))
    )
    source = GeneralPayoffMatrix(np.array([[0.0, 1.0], [1.0, 0.0]]))

    with pytest.raises(ValueError):
        approximation.approximate(source)
