from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from monocycle_nash.application.node_spec import NodeSpec, OutputSpec
from monocycle_nash.infrastructure.input.toml_matrix_config_port import TomlMatrixConfigPort


# ---------------------------------------------------------------------------
# ヘルパー
# ---------------------------------------------------------------------------


def write_toml(path: Path, content: str) -> None:
    """テキストを dedent して TOML ファイルへ書き込む。"""
    path.write_text(textwrap.dedent(content), encoding="utf-8")


# ---------------------------------------------------------------------------
# 基本的な読み込み
# ---------------------------------------------------------------------------


def test_load_simple_node_spec(tmp_path: Path) -> None:
    write_toml(tmp_path / "simple.toml", """
        method = "monocycle_from_characters"
        name = "rps"
    """)
    port = TomlMatrixConfigPort(data_dir=tmp_path)
    spec = port.load_node_spec("simple")
    assert isinstance(spec, NodeSpec)
    assert spec.method == "monocycle_from_characters"
    assert spec.name == "rps"


def test_load_node_spec_default_name(tmp_path: Path) -> None:
    """name を省略した場合は "root" になる。"""
    write_toml(tmp_path / "no_name.toml", """
        method = "general_from_raw"
    """)
    port = TomlMatrixConfigPort(data_dir=tmp_path)
    spec = port.load_node_spec("no_name")
    assert spec.name == "root"


def test_load_node_spec_with_absolute_path(tmp_path: Path) -> None:
    toml_path = tmp_path / "abs.toml"
    write_toml(toml_path, """
        method = "general_from_raw"
    """)
    port = TomlMatrixConfigPort(data_dir=tmp_path)
    spec = port.load_node_spec(str(toml_path))
    assert spec.method == "general_from_raw"


# ---------------------------------------------------------------------------
# params の読み込み
# ---------------------------------------------------------------------------


def test_load_node_spec_with_params(tmp_path: Path) -> None:
    write_toml(tmp_path / "params.toml", """
        method = "random_skew_symmetric"
        name = "rand"

        [params]
        size = 3
        low = -2.0
        high = 2.0
        seed = 42
    """)
    port = TomlMatrixConfigPort(data_dir=tmp_path)
    spec = port.load_node_spec("params")
    assert spec.params["size"] == 3
    assert spec.params["seed"] == 42
    assert spec.params["low"] == pytest.approx(-2.0)


def test_load_node_spec_inline_characters(tmp_path: Path) -> None:
    """[[params.characters]] でインライン定義したキャラクターが params に入る。"""
    write_toml(tmp_path / "inline.toml", """
        method = "monocycle_from_characters"

        [[params.characters]]
        power = 1.0
        vector = [1.0, 0.0]
        label = "Rock"

        [[params.characters]]
        power = 0.5
        vector = [0.0, 1.0]
        label = "Paper"
    """)
    port = TomlMatrixConfigPort(data_dir=tmp_path)
    spec = port.load_node_spec("inline")
    chars = spec.params["characters"]
    assert len(chars) == 2
    assert chars[0]["label"] == "Rock"
    assert chars[1]["label"] == "Paper"


# ---------------------------------------------------------------------------
# refs の読み込み
# ---------------------------------------------------------------------------


def test_load_node_spec_with_refs(tmp_path: Path) -> None:
    write_toml(tmp_path / "refs.toml", """
        method = "monocycle_from_characters"

        [refs]
        characters = "path/to/chars.toml"
    """)
    port = TomlMatrixConfigPort(data_dir=tmp_path)
    spec = port.load_node_spec("refs")
    assert spec.refs["characters"] == "path/to/chars.toml"


def test_load_node_spec_refs_values_are_strings(tmp_path: Path) -> None:
    """refs の値は常に str に変換される。"""
    write_toml(tmp_path / "refs_str.toml", """
        method = "monocycle_from_characters"

        [refs]
        characters = "chars"
    """)
    port = TomlMatrixConfigPort(data_dir=tmp_path)
    spec = port.load_node_spec("refs_str")
    assert isinstance(spec.refs["characters"], str)


# ---------------------------------------------------------------------------
# children の読み込み
# ---------------------------------------------------------------------------


def test_load_node_spec_with_children(tmp_path: Path) -> None:
    write_toml(tmp_path / "children.toml", """
        method = "general_from_team_matchups"
        name = "team"

        [refs]
        teams = "teams.toml"

        [children.character_matrix]
        method = "monocycle_from_characters"
        name = "chars"

        [children.character_matrix.refs]
        characters = "chars.toml"
    """)
    port = TomlMatrixConfigPort(data_dir=tmp_path)
    spec = port.load_node_spec("children")
    assert "character_matrix" in spec.children
    child = spec.children["character_matrix"]
    assert isinstance(child, NodeSpec)
    assert child.method == "monocycle_from_characters"
    assert child.name == "chars"
    assert child.refs["characters"] == "chars.toml"


def test_load_node_spec_deeply_nested_children(tmp_path: Path) -> None:
    """2段階ネストした children が再帰的に解析される。"""
    write_toml(tmp_path / "deep.toml", """
        method = "approx_monocycle_to_general"
        name = "approx"

        [children.source]
        method = "monocycle_from_characters"

        [children.source.refs]
        characters = "chars.toml"
    """)
    port = TomlMatrixConfigPort(data_dir=tmp_path)
    spec = port.load_node_spec("deep")
    source = spec.children["source"]
    assert source.method == "monocycle_from_characters"
    assert source.refs["characters"] == "chars.toml"


# ---------------------------------------------------------------------------
# outputs の読み込み
# ---------------------------------------------------------------------------


def test_load_node_spec_with_outputs(tmp_path: Path) -> None:
    write_toml(tmp_path / "outputs.toml", """
        method = "monocycle_from_characters"

        [[outputs]]
        method = "payoff_directed_graph"
        [outputs.params]
        filename = "graph.svg"
        threshold = 0.1
    """)
    port = TomlMatrixConfigPort(data_dir=tmp_path)
    spec = port.load_node_spec("outputs")
    assert len(spec.outputs) == 1
    out = spec.outputs[0]
    assert isinstance(out, OutputSpec)
    assert out.method == "payoff_directed_graph"
    assert out.params["filename"] == "graph.svg"
    assert out.params["threshold"] == pytest.approx(0.1)


def test_load_node_spec_multiple_outputs(tmp_path: Path) -> None:
    write_toml(tmp_path / "multi_out.toml", """
        method = "monocycle_from_characters"

        [[outputs]]
        method = "payoff_directed_graph"

        [[outputs]]
        method = "character_vector_graph"
    """)
    port = TomlMatrixConfigPort(data_dir=tmp_path)
    spec = port.load_node_spec("multi_out")
    assert len(spec.outputs) == 2
    methods = [o.method for o in spec.outputs]
    assert "payoff_directed_graph" in methods
    assert "character_vector_graph" in methods


def test_load_node_spec_no_outputs_gives_empty_tuple(tmp_path: Path) -> None:
    write_toml(tmp_path / "no_out.toml", """
        method = "monocycle_from_characters"
    """)
    port = TomlMatrixConfigPort(data_dir=tmp_path)
    spec = port.load_node_spec("no_out")
    assert spec.outputs == ()


# ---------------------------------------------------------------------------
# エラーケース
# ---------------------------------------------------------------------------


def test_load_node_spec_file_not_found(tmp_path: Path) -> None:
    port = TomlMatrixConfigPort(data_dir=tmp_path)
    with pytest.raises(FileNotFoundError, match="設定ファイルが見つかりません"):
        port.load_node_spec("nonexistent")


def test_load_node_spec_missing_method_raises(tmp_path: Path) -> None:
    write_toml(tmp_path / "no_method.toml", """
        name = "no_method"
    """)
    port = TomlMatrixConfigPort(data_dir=tmp_path)
    with pytest.raises(KeyError, match="method"):
        port.load_node_spec("no_method")


def test_load_node_spec_output_missing_method_raises(tmp_path: Path) -> None:
    write_toml(tmp_path / "bad_out.toml", """
        method = "monocycle_from_characters"

        [[outputs]]
        filename = "out.svg"
    """)
    port = TomlMatrixConfigPort(data_dir=tmp_path)
    with pytest.raises(KeyError, match="method"):
        port.load_node_spec("bad_out")


# ---------------------------------------------------------------------------
# config_id のパス解決
# ---------------------------------------------------------------------------


def test_config_id_with_extension(tmp_path: Path) -> None:
    write_toml(tmp_path / "ext.toml", """
        method = "general_from_raw"
    """)
    port = TomlMatrixConfigPort(data_dir=tmp_path)
    spec = port.load_node_spec("ext.toml")
    assert spec.method == "general_from_raw"


def test_config_id_without_extension(tmp_path: Path) -> None:
    write_toml(tmp_path / "noext.toml", """
        method = "general_from_raw"
    """)
    port = TomlMatrixConfigPort(data_dir=tmp_path)
    spec = port.load_node_spec("noext")
    assert spec.method == "general_from_raw"
