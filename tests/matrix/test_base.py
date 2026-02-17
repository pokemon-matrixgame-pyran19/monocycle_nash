import numpy as np
import pytest

from monocycle_nash.matrix.general import GeneralPayoffMatrix


def test_eigenvalues_returns_imaginary_magnitudes_for_alternating_matrix() -> None:
    matrix = np.array([[0.0, 2.0], [-2.0, 0.0]], dtype=float)
    payoff = GeneralPayoffMatrix(matrix)

    values = payoff.eigenvalues()

    np.testing.assert_allclose(np.sort(values), np.array([2.0, 2.0]))


def test_eigenvalues_raise_for_non_alternating_matrix() -> None:
    matrix = np.array([[2.0, 0.0], [0.0, -1.0]], dtype=float)
    payoff = GeneralPayoffMatrix(matrix)

    with pytest.raises(ValueError, match="交代行列"):
        payoff.eigenvalues()
