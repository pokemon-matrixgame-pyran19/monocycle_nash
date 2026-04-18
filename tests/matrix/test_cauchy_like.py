"""
CauchyLikePayoffMatrix のテスト

コーシー型行列 B_ij = a_i * a_j / (b_i - b_j) の構築・計算・均衡解を検証する。

3次元・5次元の具体的な数値例を使い、
- 行列構造の正確性（交代行列・対角0）
- 理論式との比較
- 数値均衡解との一致
を確認する。
"""

import numpy as np
import pytest

from monocycle_nash.domain.matrix.cauchy_like import (
    CauchyLikeEntity,
    CauchyLikePureStrategy,
    CauchyLikePayoffMatrix,
)
from monocycle_nash.domain.solver.selector import SolverSelector
from monocycle_nash.domain.matrix.base import PayoffMatrix


# ---------------------------------------------------------------------------
# 3次元テストパラメータ
# b = [1, 2, 3], a = [1, -1, 1] のとき:
#   v_i = Π_{j≠i}(b_i - b_j):
#     v_1 = (1-2)(1-3) = 2
#     v_2 = (2-1)(2-3) = -1
#     v_3 = (3-1)(3-2) = 2
#   x_i ∝ v_i / a_i = [2, 1, 2], 正規化 → [2/5, 1/5, 2/5]
# ---------------------------------------------------------------------------
A_3 = [1.0, -1.0, 1.0]
B_3 = [1.0, 2.0, 3.0]
EQUILIBRIUM_3 = np.array([2.0 / 5.0, 1.0 / 5.0, 2.0 / 5.0])

# ---------------------------------------------------------------------------
# 5次元テストパラメータ
# b = [1, 2, 3, 4, 5], a = [1, -1, 1, -1, 1] のとき:
#   均衡確率は M (Cauchy行列) の零ベクトル u から x_j = u_j / a_j で算出。
#   数値結果（SVD ベース）:
#     x ≈ [0.24920, 0.13419, 0.23323, 0.13419, 0.24920]
# ---------------------------------------------------------------------------
A_5 = [1.0, -1.0, 1.0, -1.0, 1.0]
B_5 = [1.0, 2.0, 3.0, 4.0, 5.0]
EQUILIBRIUM_5 = np.array([0.24920128, 0.1341853, 0.23322684, 0.1341853, 0.24920128])


# ===========================================================================
# CauchyLikeEntity / CauchyLikePureStrategy のテスト
# ===========================================================================


class TestCauchyLikeEntity:
    def test_creation(self) -> None:
        entity = CauchyLikeEntity(a=2.0, b=3.0, label="x")
        assert entity.a == pytest.approx(2.0)
        assert entity.b == pytest.approx(3.0)
        assert entity.label == "x"

    def test_default_label(self) -> None:
        entity = CauchyLikeEntity(a=1.0, b=0.0)
        assert entity.label == ""

    def test_frozen(self) -> None:
        entity = CauchyLikeEntity(a=1.0, b=2.0)
        with pytest.raises(Exception):
            entity.a = 99.0  # type: ignore[misc]


class TestCauchyLikePureStrategy:
    def test_a_b_properties(self) -> None:
        entity = CauchyLikeEntity(a=3.0, b=-1.0, label="s1")
        strategy = CauchyLikePureStrategy(id="s1", entity=entity)
        assert strategy.a == pytest.approx(3.0)
        assert strategy.b == pytest.approx(-1.0)
        assert strategy.label == "s1"

    def test_cast_valid(self) -> None:
        entity = CauchyLikeEntity(a=1.0, b=0.0, label="s0")
        strategy = CauchyLikePureStrategy(id="s0", entity=entity)
        assert CauchyLikePureStrategy.cast(strategy) is strategy

    def test_cast_invalid_raises(self) -> None:
        from monocycle_nash.domain.strategy import PureStrategy, LabelEntity

        non_cauchy = PureStrategy(id="s0", entity=LabelEntity(label="s0"))
        with pytest.raises(TypeError, match="CauchyLikePureStrategy"):
            CauchyLikePureStrategy.cast(non_cauchy)


# ===========================================================================
# CauchyLikePayoffMatrix のテスト
# ===========================================================================


class TestCauchyLikePayoffMatrixConstruction:
    def test_is_payoff_matrix(self) -> None:
        m = CauchyLikePayoffMatrix.from_ab_lists(A_3, B_3)
        assert isinstance(m, PayoffMatrix)

    def test_size_3d(self) -> None:
        m = CauchyLikePayoffMatrix.from_ab_lists(A_3, B_3)
        assert m.size == 3

    def test_size_5d(self) -> None:
        m = CauchyLikePayoffMatrix.from_ab_lists(A_5, B_5)
        assert m.size == 5

    def test_diagonal_zero_3d(self) -> None:
        m = CauchyLikePayoffMatrix.from_ab_lists(A_3, B_3)
        for i in range(m.size):
            assert m.matrix[i, i] == pytest.approx(0.0, abs=1e-12)

    def test_diagonal_zero_5d(self) -> None:
        m = CauchyLikePayoffMatrix.from_ab_lists(A_5, B_5)
        for i in range(m.size):
            assert m.matrix[i, i] == pytest.approx(0.0, abs=1e-12)

    def test_skew_symmetric_3d(self) -> None:
        m = CauchyLikePayoffMatrix.from_ab_lists(A_3, B_3)
        np.testing.assert_allclose(m.matrix + m.matrix.T, 0.0, atol=1e-12)

    def test_skew_symmetric_5d(self) -> None:
        m = CauchyLikePayoffMatrix.from_ab_lists(A_5, B_5)
        np.testing.assert_allclose(m.matrix + m.matrix.T, 0.0, atol=1e-12)

    def test_matrix_values_3d(self) -> None:
        """B_ij = a_i * a_j / (b_i - b_j) の具体値を検証。"""
        m = CauchyLikePayoffMatrix.from_ab_lists(A_3, B_3)
        mat = m.matrix
        # B[0,1] = a0*a1/(b0-b1) = 1*(-1)/(1-2) = 1
        assert mat[0, 1] == pytest.approx(1.0, abs=1e-12)
        # B[0,2] = a0*a2/(b0-b2) = 1*1/(1-3) = -0.5
        assert mat[0, 2] == pytest.approx(-0.5, abs=1e-12)
        # B[1,2] = a1*a2/(b1-b2) = (-1)*1/(2-3) = 1
        assert mat[1, 2] == pytest.approx(1.0, abs=1e-12)

    def test_from_params_tuple(self) -> None:
        """(a, b) タプルリストからの構築。"""
        params = [(1.0, 1.0), (-1.0, 2.0), (1.0, 3.0)]
        m = CauchyLikePayoffMatrix(params)
        assert m.size == 3

    def test_from_params_tuple_with_label(self) -> None:
        """(a, b, label) タプルリストからの構築。"""
        params = [(1.0, 1.0, "A"), (-1.0, 2.0, "B"), (1.0, 3.0, "C")]
        m = CauchyLikePayoffMatrix(params)
        assert m.labels == ["A", "B", "C"]

    def test_labels_default(self) -> None:
        m = CauchyLikePayoffMatrix.from_ab_lists(A_3, B_3)
        assert m.labels == ["s0", "s1", "s2"]

    def test_labels_custom(self) -> None:
        m = CauchyLikePayoffMatrix.from_ab_lists(A_3, B_3, labels=["X", "Y", "Z"])
        assert m.labels == ["X", "Y", "Z"]

    def test_equal_b_raises(self) -> None:
        """b_i = b_j の場合はゼロ除算でエラー。"""
        with pytest.raises(ValueError, match="b_i と b_j が等しい"):
            CauchyLikePayoffMatrix.from_ab_lists([1.0, 1.0], [1.0, 1.0])

    def test_mismatched_ab_lengths_raises(self) -> None:
        with pytest.raises(ValueError, match="長さが一致"):
            CauchyLikePayoffMatrix.from_ab_lists([1.0, 2.0], [1.0])


class TestCauchyLikePayoffMatrixParameters:
    def test_get_a_values_3d(self) -> None:
        m = CauchyLikePayoffMatrix.from_ab_lists(A_3, B_3)
        np.testing.assert_array_almost_equal(m.get_a_values(), A_3)

    def test_get_b_values_3d(self) -> None:
        m = CauchyLikePayoffMatrix.from_ab_lists(A_3, B_3)
        np.testing.assert_array_almost_equal(m.get_b_values(), B_3)

    def test_get_a_values_5d(self) -> None:
        m = CauchyLikePayoffMatrix.from_ab_lists(A_5, B_5)
        np.testing.assert_array_almost_equal(m.get_a_values(), A_5)

    def test_get_b_values_5d(self) -> None:
        m = CauchyLikePayoffMatrix.from_ab_lists(A_5, B_5)
        np.testing.assert_array_almost_equal(m.get_b_values(), B_5)


class TestProductFormula:
    """product_formula(): v_i = Π_{j≠i}(b_i - b_j) のテスト。"""

    def test_product_formula_3d(self) -> None:
        m = CauchyLikePayoffMatrix.from_ab_lists(A_3, B_3)
        v = m.product_formula()
        assert v[0] == pytest.approx(2.0, abs=1e-12)   # (1-2)(1-3) = 2
        assert v[1] == pytest.approx(-1.0, abs=1e-12)  # (2-1)(2-3) = -1
        assert v[2] == pytest.approx(2.0, abs=1e-12)   # (3-1)(3-2) = 2

    def test_product_formula_5d(self) -> None:
        m = CauchyLikePayoffMatrix.from_ab_lists(A_5, B_5)
        v = m.product_formula()
        np.testing.assert_allclose(v, [24.0, -6.0, 4.0, -6.0, 24.0], atol=1e-10)

    def test_product_formula_has_negative_values_3d(self) -> None:
        """B_3 (b=[1,2,3]) では product_formula に負の成分が含まれることを確認。

        a = [1,1,1] (全て同符号) を使用: A_3 とは異なるパラメータ。
        b = [1,2,3] のとき v_i = Π_{j≠i}(b_i - b_j) は
        v_2 = (2-1)(2-3) = -1 < 0 となる。
        このまま確率として使うことはできないため、a_i 除算と正規化が必要。
        """
        m = CauchyLikePayoffMatrix.from_ab_lists([1.0, 1.0, 1.0], B_3)
        v = m.product_formula()
        assert np.any(v < 0), (
            "b=[1,2,3] のとき vi=Π_{j≠i}(bi-bj) には負の成分が含まれるはず"
        )


class TestTheoreticalEquilibrium:
    """theoretical_equilibrium(): 正しい理論式 x_i ∝ v_i / a_i のテスト。"""

    def test_theoretical_equilibrium_3d(self) -> None:
        m = CauchyLikePayoffMatrix.from_ab_lists(A_3, B_3)
        x = m.theoretical_equilibrium()
        np.testing.assert_allclose(x, EQUILIBRIUM_3, atol=1e-12)

    def test_theoretical_equilibrium_5d(self) -> None:
        m = CauchyLikePayoffMatrix.from_ab_lists(A_5, B_5)
        x = m.theoretical_equilibrium()
        np.testing.assert_allclose(x, EQUILIBRIUM_5, atol=1e-12)

    def test_theoretical_equilibrium_sums_to_one_3d(self) -> None:
        m = CauchyLikePayoffMatrix.from_ab_lists(A_3, B_3)
        x = m.theoretical_equilibrium()
        assert x.sum() == pytest.approx(1.0, abs=1e-12)

    def test_theoretical_equilibrium_sums_to_one_5d(self) -> None:
        m = CauchyLikePayoffMatrix.from_ab_lists(A_5, B_5)
        x = m.theoretical_equilibrium()
        assert x.sum() == pytest.approx(1.0, abs=1e-12)

    def test_theoretical_equilibrium_all_positive_3d(self) -> None:
        m = CauchyLikePayoffMatrix.from_ab_lists(A_3, B_3)
        x = m.theoretical_equilibrium()
        assert np.all(x > 0)

    def test_theoretical_equilibrium_all_positive_5d(self) -> None:
        m = CauchyLikePayoffMatrix.from_ab_lists(A_5, B_5)
        x = m.theoretical_equilibrium()
        assert np.all(x > 0)

    def test_theoretical_equilibrium_sign_independent(self) -> None:
        """a = [1, 1, 1] (全正符号) でも general ソルバー経由で有効な確率分布が返る。"""
        m = CauchyLikePayoffMatrix.from_ab_lists([1.0, 1.0, 1.0], B_3)
        x = m.theoretical_equilibrium()
        assert x.sum() == pytest.approx(1.0, abs=1e-6)
        assert np.all(x >= -1e-10)

    def test_theory_satisfies_equilibrium_condition_3d(self) -> None:
        """理論式が均衡条件 Bx = 0 を満たすことを直接検証。"""
        m = CauchyLikePayoffMatrix.from_ab_lists(A_3, B_3)
        x = m.theoretical_equilibrium()
        bx = m.matrix @ x
        np.testing.assert_allclose(bx, np.zeros(3), atol=1e-12)

    def test_theory_satisfies_equilibrium_condition_5d(self) -> None:
        """理論式が均衡条件 Bx = 0 を満たすことを直接検証。"""
        m = CauchyLikePayoffMatrix.from_ab_lists(A_5, B_5)
        x = m.theoretical_equilibrium()
        bx = m.matrix @ x
        np.testing.assert_allclose(bx, np.zeros(5), atol=1e-10)


class TestSolveEquilibrium:
    """SolverSelector: 数値解が理論解と一致することを検証。"""

    def test_numerical_matches_theory_3d(self) -> None:
        m = CauchyLikePayoffMatrix.from_ab_lists(A_3, B_3)
        theory = m.theoretical_equilibrium()
        numerical = SolverSelector().solve(m)
        np.testing.assert_allclose(
            numerical.probabilities, theory, atol=1e-6,
            err_msg="3次元: 数値均衡解が理論解と一致しない"
        )

    def test_numerical_matches_theory_5d(self) -> None:
        m = CauchyLikePayoffMatrix.from_ab_lists(A_5, B_5)
        theory = m.theoretical_equilibrium()
        numerical = SolverSelector().solve(m)
        np.testing.assert_allclose(
            numerical.probabilities, theory, atol=1e-6,
            err_msg="5次元: 数値均衡解が理論解と一致しない"
        )

    def test_numerical_sums_to_one_3d(self) -> None:
        m = CauchyLikePayoffMatrix.from_ab_lists(A_3, B_3)
        eq = SolverSelector().solve(m)
        assert eq.probabilities.sum() == pytest.approx(1.0, abs=1e-6)

    def test_numerical_sums_to_one_5d(self) -> None:
        m = CauchyLikePayoffMatrix.from_ab_lists(A_5, B_5)
        eq = SolverSelector().solve(m)
        assert eq.probabilities.sum() == pytest.approx(1.0, abs=1e-6)
