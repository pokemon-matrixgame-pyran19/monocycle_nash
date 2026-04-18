from __future__ import annotations

import pytest

from monocycle_nash.application.matrix_node_factory import MatrixNodeFactory
from monocycle_nash.application.matrix_nodes import (
    ApproxDominantEigenpairNode,
    ApproxEquilibriumPreservingNode,
    ApproxMonocycleToGeneralNode,
    CharacterInlineSource,
    CharacterListFromFileNode,
    CharacterVectorGraphOutputNode,
    EquilibriumOutputNode,
    GeneralFromRawNode,
    GeneralFromTeamMatchupsNode,
    GeneralFromTeamsPayoffNode,
    MonocycleFromCharactersNode,
    PayoffDirectedGraphOutputNode,
    RandomSkewSymmetricNode,
    TeamInlineSource,
    TeamListFromFileNode,
)
from monocycle_nash.application.node_spec import NodeSpec, OutputSpec


# ---------------------------------------------------------------------------
# ヘルパー
# ---------------------------------------------------------------------------

FACTORY = MatrixNodeFactory()

INLINE_CHARS_PARAMS = {
    "characters": [
        {"power": 1.0, "vector": [1.0, 0.0], "label": "A"},
        {"power": -1.0, "vector": [0.0, 1.0], "label": "B"},
    ]
}

INLINE_TEAMS_PARAMS = {
    "teams": [
        {"label": "A+B", "member_ids": ["A", "B"]},
        {"label": "B+C", "member_ids": ["B", "C"]},
    ]
}


# ---------------------------------------------------------------------------
# GeneralFromRawNode
# ---------------------------------------------------------------------------


def test_build_general_from_raw() -> None:
    spec = NodeSpec(
        method="general_from_raw",
        name="raw",
        params={"matrix": [[0.0, 1.0], [-1.0, 0.0]], "labels": ["A", "B"]},
    )
    node = FACTORY.build(spec)
    assert isinstance(node, GeneralFromRawNode)
    assert node.name == "raw"
    assert node.labels == ["A", "B"]


def test_build_general_from_raw_no_labels() -> None:
    spec = NodeSpec(
        method="general_from_raw",
        params={"matrix": [[0.0, 1.0], [-1.0, 0.0]]},
    )
    node = FACTORY.build(spec)
    assert isinstance(node, GeneralFromRawNode)
    assert node.labels is None


# ---------------------------------------------------------------------------
# MonocycleFromCharactersNode — inline
# ---------------------------------------------------------------------------


def test_build_monocycle_inline_characters() -> None:
    spec = NodeSpec(method="monocycle_from_characters", params=INLINE_CHARS_PARAMS)
    node = FACTORY.build(spec)
    assert isinstance(node, MonocycleFromCharactersNode)
    assert isinstance(node.characters, CharacterInlineSource)
    assert len(node.characters.characters) == 2
    assert node.characters.characters[0].label == "A"


def test_build_monocycle_inline_characters_vector_cast() -> None:
    """vector 要素は float にキャストされる。"""
    spec = NodeSpec(
        method="monocycle_from_characters",
        params={"characters": [{"power": 1, "vector": [1, 0], "label": "X"}]},
    )
    node = FACTORY.build(spec)
    assert isinstance(node.characters, CharacterInlineSource)
    v = node.characters.characters[0].vector
    assert v == (1.0, 0.0)
    assert isinstance(v[0], float)


# ---------------------------------------------------------------------------
# MonocycleFromCharactersNode — refs
# ---------------------------------------------------------------------------


def test_build_monocycle_file_ref_characters() -> None:
    spec = NodeSpec(
        method="monocycle_from_characters",
        refs={"characters": "path/to/chars.toml"},
    )
    node = FACTORY.build(spec)
    assert isinstance(node, MonocycleFromCharactersNode)
    assert isinstance(node.characters, CharacterListFromFileNode)
    assert node.characters.path == "path/to/chars.toml"


def test_build_monocycle_refs_takes_priority_over_params() -> None:
    """refs と params の両方がある場合は refs が優先される。"""
    spec = NodeSpec(
        method="monocycle_from_characters",
        params=INLINE_CHARS_PARAMS,
        refs={"characters": "override.toml"},
    )
    node = FACTORY.build(spec)
    assert isinstance(node.characters, CharacterListFromFileNode)


# ---------------------------------------------------------------------------
# GeneralFromTeamsPayoffNode
# ---------------------------------------------------------------------------


def test_build_general_from_teams_payoff_inline() -> None:
    spec = NodeSpec(
        method="general_from_teams_payoff",
        params={"team_payoff": [[0.0, 1.0], [-1.0, 0.0]], **INLINE_TEAMS_PARAMS},
    )
    node = FACTORY.build(spec)
    assert isinstance(node, GeneralFromTeamsPayoffNode)
    assert isinstance(node.teams, TeamInlineSource)


def test_build_general_from_teams_payoff_file_ref() -> None:
    spec = NodeSpec(
        method="general_from_teams_payoff",
        params={"team_payoff": [[0.0, 1.0], [-1.0, 0.0]]},
        refs={"teams": "teams.toml"},
    )
    node = FACTORY.build(spec)
    assert isinstance(node, GeneralFromTeamsPayoffNode)
    assert isinstance(node.teams, TeamListFromFileNode)
    assert node.teams.path == "teams.toml"


# ---------------------------------------------------------------------------
# GeneralFromTeamMatchupsNode
# ---------------------------------------------------------------------------


def test_build_general_from_team_matchups() -> None:
    char_spec = NodeSpec(method="monocycle_from_characters", params=INLINE_CHARS_PARAMS)
    spec = NodeSpec(
        method="general_from_team_matchups",
        params={**INLINE_TEAMS_PARAMS, "use_monocycle_formula": False},
        children={"character_matrix": char_spec},
    )
    node = FACTORY.build(spec)
    assert isinstance(node, GeneralFromTeamMatchupsNode)
    assert isinstance(node.teams, TeamInlineSource)
    assert isinstance(node.character_matrix, MonocycleFromCharactersNode)
    assert node.use_monocycle_formula is False


def test_build_general_from_team_matchups_missing_child_raises() -> None:
    spec = NodeSpec(
        method="general_from_team_matchups",
        params=INLINE_TEAMS_PARAMS,
    )
    with pytest.raises(ValueError, match="character_matrix"):
        FACTORY.build(spec)


def test_build_general_from_team_matchups_default_monocycle_formula() -> None:
    char_spec = NodeSpec(method="monocycle_from_characters", params=INLINE_CHARS_PARAMS)
    spec = NodeSpec(
        method="general_from_team_matchups",
        params=INLINE_TEAMS_PARAMS,
        children={"character_matrix": char_spec},
    )
    node = FACTORY.build(spec)
    assert node.use_monocycle_formula is True


# ---------------------------------------------------------------------------
# RandomSkewSymmetricNode
# ---------------------------------------------------------------------------


def test_build_random_skew_symmetric() -> None:
    spec = NodeSpec(
        method="random_skew_symmetric",
        params={"size": 3, "low": -2.0, "high": 2.0, "seed": 99, "max_attempts": 500},
    )
    node = FACTORY.build(spec)
    assert isinstance(node, RandomSkewSymmetricNode)
    assert node.size == 3
    assert node.seed == 99
    assert node.max_attempts == 500


def test_build_random_skew_symmetric_defaults() -> None:
    spec = NodeSpec(method="random_skew_symmetric", params={"size": 2})
    node = FACTORY.build(spec)
    assert isinstance(node, RandomSkewSymmetricNode)
    assert node.low == -1.0
    assert node.high == 1.0
    assert node.seed is None
    assert node.max_attempts == 10_000


# ---------------------------------------------------------------------------
# 近似変換ノード
# ---------------------------------------------------------------------------


def _monocycle_child_spec() -> NodeSpec:
    return NodeSpec(method="monocycle_from_characters", params=INLINE_CHARS_PARAMS)


def test_build_approx_monocycle_to_general() -> None:
    spec = NodeSpec(
        method="approx_monocycle_to_general",
        children={"source": _monocycle_child_spec()},
    )
    node = FACTORY.build(spec)
    assert isinstance(node, ApproxMonocycleToGeneralNode)
    assert isinstance(node.source, MonocycleFromCharactersNode)


def test_build_approx_monocycle_to_general_missing_source_raises() -> None:
    spec = NodeSpec(method="approx_monocycle_to_general")
    with pytest.raises(ValueError, match="source"):
        FACTORY.build(spec)


def test_build_approx_dominant_eigenpair() -> None:
    spec = NodeSpec(
        method="approx_dominant_eigenpair",
        params={"atol": 1e-6},
        children={"source": _monocycle_child_spec()},
    )
    node = FACTORY.build(spec)
    assert isinstance(node, ApproxDominantEigenpairNode)
    assert node.atol == pytest.approx(1e-6)


def test_build_approx_dominant_eigenpair_default_atol() -> None:
    spec = NodeSpec(
        method="approx_dominant_eigenpair",
        children={"source": _monocycle_child_spec()},
    )
    node = FACTORY.build(spec)
    assert node.atol == pytest.approx(1e-8)


def test_build_approx_equilibrium_preserving() -> None:
    spec = NodeSpec(
        method="approx_equilibrium_preserving",
        children={"source": _monocycle_child_spec()},
    )
    node = FACTORY.build(spec)
    assert isinstance(node, ApproxEquilibriumPreservingNode)


# ---------------------------------------------------------------------------
# 出力ノード生成
# ---------------------------------------------------------------------------


def test_build_with_payoff_directed_graph_output() -> None:
    out = OutputSpec(
        method="payoff_directed_graph",
        params={"filename": "out.svg", "threshold": 0.2, "canvas_size": 600},
    )
    spec = NodeSpec(
        method="general_from_raw",
        params={"matrix": [[0.0, 1.0], [-1.0, 0.0]]},
        outputs=(out,),
    )
    node = FACTORY.build(spec)
    assert len(node.outputs) == 1
    output_node = node.outputs[0]
    assert isinstance(output_node, PayoffDirectedGraphOutputNode)
    assert output_node.filename == "out.svg"
    assert output_node.threshold == pytest.approx(0.2)
    assert output_node.canvas_size == 600


def test_build_with_character_vector_graph_output() -> None:
    out = OutputSpec(
        method="character_vector_graph",
        params={"filename": "chars.svg", "canvas_size": 720, "margin": 60},
    )
    spec = NodeSpec(
        method="monocycle_from_characters",
        params=INLINE_CHARS_PARAMS,
        outputs=(out,),
    )
    node = FACTORY.build(spec)
    assert len(node.outputs) == 1
    output_node = node.outputs[0]
    assert isinstance(output_node, CharacterVectorGraphOutputNode)
    assert output_node.margin == 60


def test_build_with_equilibrium_output() -> None:
    out = OutputSpec(
        method="equilibrium",
        params={"filename": "equilibrium_out.toml"},
    )
    spec = NodeSpec(
        method="general_from_raw",
        params={"matrix": [[0.0, 1.0], [-1.0, 0.0]]},
        outputs=(out,),
    )
    node = FACTORY.build(spec)
    assert len(node.outputs) == 1
    output_node = node.outputs[0]
    assert isinstance(output_node, EquilibriumOutputNode)
    assert output_node.filename == "equilibrium_out.toml"


def test_build_with_output_defaults() -> None:
    """output params を省略した場合にデフォルト値が適用される。"""
    out = OutputSpec(method="payoff_directed_graph")
    spec = NodeSpec(
        method="general_from_raw",
        params={"matrix": [[0.0, 1.0], [-1.0, 0.0]]},
        outputs=(out,),
    )
    node = FACTORY.build(spec)
    output_node = node.outputs[0]
    assert isinstance(output_node, PayoffDirectedGraphOutputNode)
    assert output_node.filename == "payoff_directed_graph.svg"
    assert output_node.threshold == pytest.approx(0.0)
    assert output_node.canvas_size == 840


# ---------------------------------------------------------------------------
# エラーケース
# ---------------------------------------------------------------------------


def test_build_unknown_method_raises() -> None:
    spec = NodeSpec(method="unknown_method")
    with pytest.raises(ValueError, match="未知の method"):
        FACTORY.build(spec)


def test_build_unknown_output_method_raises() -> None:
    spec = NodeSpec(
        method="general_from_raw",
        params={"matrix": [[0.0, 1.0], [-1.0, 0.0]]},
        outputs=(OutputSpec(method="unknown_output"),),
    )
    with pytest.raises(ValueError, match="未知の output method"):
        FACTORY.build(spec)


# ---------------------------------------------------------------------------
# node_name と outputs の引き継ぎ
# ---------------------------------------------------------------------------


def test_build_preserves_node_name() -> None:
    spec = NodeSpec(
        method="monocycle_from_characters",
        name="janken",
        params=INLINE_CHARS_PARAMS,
    )
    node = FACTORY.build(spec)
    assert node.name == "janken"


def test_build_no_outputs_gives_empty_tuple() -> None:
    spec = NodeSpec(
        method="monocycle_from_characters",
        params=INLINE_CHARS_PARAMS,
    )
    node = FACTORY.build(spec)
    assert node.outputs == ()
