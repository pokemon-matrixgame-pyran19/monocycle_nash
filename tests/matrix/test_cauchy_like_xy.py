import numpy as np
import pytest

from monocycle_nash.domain.matrix.base import PayoffMatrix
from monocycle_nash.domain.matrix.cauchy_like_xy import (
    CauchyLikeXYEntity,
    CauchyLikeXYPureStrategy,
    CauchyLikeXYPayoffMatrix,
)


X_3 = [1.0, 0.0, 1.0]
Y_3 = [0.0, 1.0, 1.0]


class TestCauchyLikeXYEntity:
    def test_creation(self) -> None:
        entity = CauchyLikeXYEntity(x=2.0, y=3.0, label="x")
        assert entity.x == pytest.approx(2.0)
        assert entity.y == pytest.approx(3.0)
        assert entity.label == "x"

    def test_default_label(self) -> None:
        entity = CauchyLikeXYEntity(x=1.0, y=0.0)
        assert entity.label == ""

    def test_frozen(self) -> None:
        entity = CauchyLikeXYEntity(x=1.0, y=2.0)
        with pytest.raises(Exception):
            entity.x = 99.0  # type: ignore[misc]


class TestCauchyLikeXYPureStrategy:
    def test_x_y_properties(self) -> None:
        entity = CauchyLikeXYEntity(x=3.0, y=-1.0, label="s1")
        strategy = CauchyLikeXYPureStrategy(id="s1", entity=entity)
        assert strategy.x == pytest.approx(3.0)
        assert strategy.y == pytest.approx(-1.0)
        assert strategy.label == "s1"

    def test_cast_valid(self) -> None:
        entity = CauchyLikeXYEntity(x=1.0, y=0.0, label="s0")
        strategy = CauchyLikeXYPureStrategy(id="s0", entity=entity)
        assert CauchyLikeXYPureStrategy.cast(strategy) is strategy

    def test_cast_invalid_raises(self) -> None:
        from monocycle_nash.domain.strategy import LabelEntity, PureStrategy

        non_cauchy_xy = PureStrategy(id="s0", entity=LabelEntity(label="s0"))
        with pytest.raises(TypeError, match="CauchyLikeXYPureStrategy"):
            CauchyLikeXYPureStrategy.cast(non_cauchy_xy)


class TestCauchyLikeXYPayoffMatrix:
    def test_is_payoff_matrix(self) -> None:
        m = CauchyLikeXYPayoffMatrix.from_xy_lists(X_3, Y_3)
        assert isinstance(m, PayoffMatrix)

    def test_size_3d(self) -> None:
        m = CauchyLikeXYPayoffMatrix.from_xy_lists(X_3, Y_3)
        assert m.size == 3

    def test_diagonal_zero(self) -> None:
        m = CauchyLikeXYPayoffMatrix.from_xy_lists(X_3, Y_3)
        np.testing.assert_allclose(np.diag(m.matrix), 0.0, atol=1e-12)

    def test_skew_symmetric(self) -> None:
        m = CauchyLikeXYPayoffMatrix.from_xy_lists(X_3, Y_3)
        np.testing.assert_allclose(m.matrix + m.matrix.T, 0.0, atol=1e-12)

    def test_matrix_values_3d(self) -> None:
        m = CauchyLikeXYPayoffMatrix.from_xy_lists(X_3, Y_3)
        mat = m.matrix
        assert mat[0, 1] == pytest.approx(1.0, abs=1e-12)
        assert mat[0, 2] == pytest.approx(1.0, abs=1e-12)
        assert mat[1, 2] == pytest.approx(-1.0, abs=1e-12)

    def test_from_params_tuple(self) -> None:
        params = [(1.0, 0.0), (0.0, 1.0), (1.0, 1.0)]
        m = CauchyLikeXYPayoffMatrix(params)
        assert m.size == 3

    def test_from_params_tuple_with_label(self) -> None:
        params = [(1.0, 0.0, "A"), (0.0, 1.0, "B"), (1.0, 1.0, "C")]
        m = CauchyLikeXYPayoffMatrix(params)
        assert m.labels == ["A", "B", "C"]

    def test_labels_default(self) -> None:
        m = CauchyLikeXYPayoffMatrix.from_xy_lists(X_3, Y_3)
        assert m.labels == ["s0", "s1", "s2"]

    def test_labels_custom(self) -> None:
        m = CauchyLikeXYPayoffMatrix.from_xy_lists(X_3, Y_3, labels=["X", "Y", "Z"])
        assert m.labels == ["X", "Y", "Z"]

    def test_zero_denom_raises(self) -> None:
        with pytest.raises(ValueError, match="行列を構成できません"):
            CauchyLikeXYPayoffMatrix.from_xy_lists([1.0, 2.0], [2.0, 4.0])

    def test_mismatched_xy_lengths_raises(self) -> None:
        with pytest.raises(ValueError, match="長さが一致"):
            CauchyLikeXYPayoffMatrix.from_xy_lists([1.0, 2.0], [1.0])

    def test_get_x_values(self) -> None:
        m = CauchyLikeXYPayoffMatrix.from_xy_lists(X_3, Y_3)
        np.testing.assert_array_almost_equal(m.get_x_values(), X_3)

    def test_get_y_values(self) -> None:
        m = CauchyLikeXYPayoffMatrix.from_xy_lists(X_3, Y_3)
        np.testing.assert_array_almost_equal(m.get_y_values(), Y_3)

