from __future__ import annotations

import pytest

from monocycle_nash.application.node_spec import NodeSpec, OutputSpec


# ---------------------------------------------------------------------------
# OutputSpec
# ---------------------------------------------------------------------------


def test_output_spec_required_field() -> None:
    """method は必須フィールド。"""
    spec = OutputSpec(method="payoff_directed_graph")
    assert spec.method == "payoff_directed_graph"


def test_output_spec_default_params() -> None:
    """params のデフォルトは空辞書。"""
    spec = OutputSpec(method="character_vector_graph")
    assert spec.params == {}
    assert spec.runner is None


def test_output_spec_with_params() -> None:
    spec = OutputSpec(method="payoff_directed_graph", params={"filename": "out.svg", "threshold": 0.1})
    assert spec.params["filename"] == "out.svg"
    assert spec.params["threshold"] == pytest.approx(0.1)


def test_output_spec_with_runner() -> None:
    spec = OutputSpec(method="equilibrium", runner="final")
    assert spec.runner == "final"


# ---------------------------------------------------------------------------
# NodeSpec — デフォルト値
# ---------------------------------------------------------------------------


def test_node_spec_method_only() -> None:
    """method だけ指定した場合のデフォルト値を確認する。"""
    spec = NodeSpec(method="monocycle_from_characters")
    assert spec.method == "monocycle_from_characters"
    assert spec.name == "root"
    assert spec.params == {}
    assert spec.refs == {}
    assert spec.children == {}
    assert spec.outputs == ()


def test_node_spec_custom_name() -> None:
    spec = NodeSpec(method="general_from_raw", name="my-node")
    assert spec.name == "my-node"


# ---------------------------------------------------------------------------
# NodeSpec — 各フィールドの代入
# ---------------------------------------------------------------------------


def test_node_spec_with_params() -> None:
    spec = NodeSpec(
        method="random_skew_symmetric",
        params={"size": 4, "low": -2.0, "high": 2.0, "seed": 42},
    )
    assert spec.params["size"] == 4
    assert spec.params["seed"] == 42


def test_node_spec_with_refs() -> None:
    spec = NodeSpec(
        method="monocycle_from_characters",
        refs={"characters": "path/to/chars.toml"},
    )
    assert spec.refs["characters"] == "path/to/chars.toml"


def test_node_spec_with_children() -> None:
    child = NodeSpec(method="monocycle_from_characters", name="chars")
    spec = NodeSpec(
        method="general_from_team_matchups",
        children={"character_matrix": child},
    )
    assert spec.children["character_matrix"] is child


def test_node_spec_with_outputs() -> None:
    out = OutputSpec(method="payoff_directed_graph", params={"filename": "g.svg"})
    spec = NodeSpec(method="general_from_raw", outputs=(out,))
    assert len(spec.outputs) == 1
    assert spec.outputs[0].method == "payoff_directed_graph"


def test_node_spec_all_fields() -> None:
    """全フィールドを指定した場合に正しく保持されることを確認する。"""
    child = NodeSpec(method="monocycle_from_characters", name="child")
    out = OutputSpec(method="payoff_directed_graph", params={"filename": "x.svg"})
    spec = NodeSpec(
        method="general_from_team_matchups",
        name="parent",
        params={"use_monocycle_formula": True},
        refs={"teams": "teams.toml"},
        children={"character_matrix": child},
        outputs=(out,),
    )
    assert spec.method == "general_from_team_matchups"
    assert spec.name == "parent"
    assert spec.params["use_monocycle_formula"] is True
    assert spec.refs["teams"] == "teams.toml"
    assert spec.children["character_matrix"].name == "child"
    assert spec.outputs[0].params["filename"] == "x.svg"


# ---------------------------------------------------------------------------
# NodeSpec — 独立したデフォルト値（共有されないことを確認）
# ---------------------------------------------------------------------------


def test_node_spec_default_dicts_are_independent() -> None:
    """デフォルトの params/refs/children は各インスタンスで独立している。"""
    a = NodeSpec(method="m")
    b = NodeSpec(method="m")
    a.params["key"] = "value"
    assert "key" not in b.params
