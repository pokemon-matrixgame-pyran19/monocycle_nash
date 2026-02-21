import numpy as np
import pytest

from monocycle_nash.character.domain import Character, MatchupVector
from monocycle_nash.matrix.approximation import (
    ApproximationQualityEvaluator,
    MaxElementDifferenceDistance,
    MonocycleToGeneralApproximation,
)
from monocycle_nash.matrix.builder import PayoffMatrixBuilder
from monocycle_nash.matrix.general import GeneralPayoffMatrix


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
