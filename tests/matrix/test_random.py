import numpy as np
import pytest

from monocycle_nash.matrix import (
    RandomMatrixAcceptanceCondition,
    generate_random_skew_symmetric_matrix,
)
from monocycle_nash.matrix.builder import PayoffMatrixBuilder


class RankFourCondition(RandomMatrixAcceptanceCondition):
    def is_satisfied(self, matrix: np.ndarray) -> bool:
        return int(np.linalg.matrix_rank(matrix)) == 4


class NeverPassCondition(RandomMatrixAcceptanceCondition):
    def is_satisfied(self, matrix: np.ndarray) -> bool:
        return False


def test_generate_random_skew_symmetric_matrix_defaults():
    rng = np.random.default_rng(42)
    matrix = generate_random_skew_symmetric_matrix(size=6, rng=rng)

    assert matrix.shape == (6, 6)
    np.testing.assert_allclose(np.diag(matrix), 0.0)
    np.testing.assert_allclose(matrix + matrix.T, 0.0, atol=1e-12)
    assert np.all(matrix[np.triu_indices(6, k=1)] >= -1.0)
    assert np.all(matrix[np.triu_indices(6, k=1)] <= 1.0)


def test_generate_random_skew_symmetric_matrix_with_acceptance_condition():
    rng = np.random.default_rng(123)
    matrix = generate_random_skew_symmetric_matrix(
        size=5,
        low=-0.5,
        high=0.5,
        acceptance_condition=RankFourCondition(),
        rng=rng,
        max_attempts=20_000,
    )

    assert int(np.linalg.matrix_rank(matrix)) == 4


def test_generate_random_skew_symmetric_matrix_raises_when_condition_not_met():
    with pytest.raises(RuntimeError):
        generate_random_skew_symmetric_matrix(
            size=3,
            acceptance_condition=NeverPassCondition(),
            max_attempts=10,
        )


def test_builder_from_random_matrix_creates_general_payoff_matrix():
    matrix = PayoffMatrixBuilder.from_random_matrix(size=4)

    assert matrix.matrix.shape == (4, 4)
    np.testing.assert_allclose(np.diag(matrix.matrix), 0.0)
    np.testing.assert_allclose(matrix.matrix + matrix.matrix.T, 0.0, atol=1e-12)
