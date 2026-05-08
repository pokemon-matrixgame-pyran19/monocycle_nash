from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from monocycle_nash.application.node_spec import NodeSpec, OutputSpec
from monocycle_nash.application.snapshot import ConfigTreeSnapshot
from monocycle_nash.infrastructure.output.toml_snapshot_store import (
    TomlConfigTreeSnapshotStore,
)


# ---------------------------------------------------------------------------
# 保存先パス
# ---------------------------------------------------------------------------


def test_store_creates_file_at_expected_path(tmp_path: Path) -> None:
    """store() が result/<run_id>/input/config_tree.toml を作成する。"""
    store = TomlConfigTreeSnapshotStore(result_base_dir=tmp_path / "result")
    snapshot = ConfigTreeSnapshot(root=NodeSpec(method="general_from_raw"))

    path = store.store("42", snapshot)

    assert path == tmp_path / "result" / "42" / "input" / "config_tree.toml"
    assert path.exists()


def test_store_creates_parent_dirs(tmp_path: Path) -> None:
    """store() は親ディレクトリが存在しなくても自動作成する。"""
    store = TomlConfigTreeSnapshotStore(result_base_dir=tmp_path / "deep" / "result")
    snapshot = ConfigTreeSnapshot(root=NodeSpec(method="general_from_raw"))

    path = store.store("1", snapshot)

    assert path.parent.is_dir()


# ---------------------------------------------------------------------------
# TOML 内容 — ラウンドトリップ
# ---------------------------------------------------------------------------


def test_store_roundtrip_simple_spec(tmp_path: Path) -> None:
    """単純な NodeSpec を保存・再読み込みすると元の内容と一致する。"""
    spec = NodeSpec(
        method="monocycle_from_characters",
        name="rps",
    )
    store = TomlConfigTreeSnapshotStore(result_base_dir=tmp_path)
    path = store.store("1", ConfigTreeSnapshot(root=spec))

    with path.open("rb") as f:
        data = tomllib.load(f)

    assert data["method"] == "monocycle_from_characters"
    assert data["name"] == "rps"


def test_store_roundtrip_with_params(tmp_path: Path) -> None:
    """params を含む NodeSpec が正しく保存される。"""
    spec = NodeSpec(
        method="random_skew_symmetric",
        name="rand",
        params={"size": 3, "low": -1.0, "high": 1.0, "seed": 42},
    )
    store = TomlConfigTreeSnapshotStore(result_base_dir=tmp_path)
    path = store.store("1", ConfigTreeSnapshot(root=spec))

    with path.open("rb") as f:
        data = tomllib.load(f)

    assert data["params"]["size"] == 3
    assert data["params"]["seed"] == 42
    assert data["params"]["low"] == pytest.approx(-1.0)


def test_store_roundtrip_with_refs(tmp_path: Path) -> None:
    """refs を含む NodeSpec が正しく保存される。"""
    spec = NodeSpec(
        method="monocycle_from_characters",
        refs={"characters": "path/to/chars.toml"},
    )
    store = TomlConfigTreeSnapshotStore(result_base_dir=tmp_path)
    path = store.store("1", ConfigTreeSnapshot(root=spec))

    with path.open("rb") as f:
        data = tomllib.load(f)

    assert data["refs"]["characters"] == "path/to/chars.toml"


def test_store_roundtrip_with_children(tmp_path: Path) -> None:
    """children を含む NodeSpec が再帰的に保存される。"""
    child = NodeSpec(
        method="monocycle_from_characters",
        name="chars",
        refs={"characters": "chars.toml"},
    )
    spec = NodeSpec(
        method="general_from_team_matchups",
        name="team",
        children={"character_matrix": child},
    )
    store = TomlConfigTreeSnapshotStore(result_base_dir=tmp_path)
    path = store.store("1", ConfigTreeSnapshot(root=spec))

    with path.open("rb") as f:
        data = tomllib.load(f)

    child_data = data["children"]["character_matrix"]
    assert child_data["method"] == "monocycle_from_characters"
    assert child_data["name"] == "chars"
    assert child_data["refs"]["characters"] == "chars.toml"


def test_store_roundtrip_with_outputs(tmp_path: Path) -> None:
    """outputs を含む NodeSpec が正しく保存される。"""
    spec = NodeSpec(
        method="monocycle_from_characters",
        outputs=(
            OutputSpec(
                method="payoff_directed_graph",
                runner="final",
                params={"filename": "graph.svg", "threshold": 0.1},
            ),
        ),
    )
    store = TomlConfigTreeSnapshotStore(result_base_dir=tmp_path)
    path = store.store("1", ConfigTreeSnapshot(root=spec))

    with path.open("rb") as f:
        data = tomllib.load(f)

    outputs = data["outputs"]
    assert len(outputs) == 1
    assert outputs[0]["method"] == "payoff_directed_graph"
    assert outputs[0]["runner"] == "final"
    assert outputs[0]["params"]["filename"] == "graph.svg"
    assert outputs[0]["params"]["threshold"] == pytest.approx(0.1)


def test_store_omits_empty_sections(tmp_path: Path) -> None:
    """params/refs/children/outputs が空の場合はキーが出力されない。"""
    spec = NodeSpec(method="general_from_raw", name="root")
    store = TomlConfigTreeSnapshotStore(result_base_dir=tmp_path)
    path = store.store("1", ConfigTreeSnapshot(root=spec))

    with path.open("rb") as f:
        data = tomllib.load(f)

    assert "params" not in data
    assert "refs" not in data
    assert "children" not in data
    assert "outputs" not in data


def test_store_inline_characters_roundtrip(tmp_path: Path) -> None:
    """インラインキャラクター定義（リスト of dict）が正しく保存・再読み込みされる。"""
    spec = NodeSpec(
        method="monocycle_from_characters",
        params={
            "characters": [
                {"power": 1.0, "vector": [1.0, 0.0], "label": "Rock"},
                {"power": 0.0, "vector": [0.0, 1.0], "label": "Paper"},
            ]
        },
    )
    store = TomlConfigTreeSnapshotStore(result_base_dir=tmp_path)
    path = store.store("1", ConfigTreeSnapshot(root=spec))

    with path.open("rb") as f:
        data = tomllib.load(f)

    chars = data["params"]["characters"]
    assert len(chars) == 2
    assert chars[0]["label"] == "Rock"
    assert chars[1]["label"] == "Paper"


def test_store_returns_path(tmp_path: Path) -> None:
    """store() は保存先の Path オブジェクトを返す。"""
    store = TomlConfigTreeSnapshotStore(result_base_dir=tmp_path)
    snapshot = ConfigTreeSnapshot(root=NodeSpec(method="general_from_raw"))

    result = store.store("99", snapshot)

    assert isinstance(result, Path)
    assert result.suffix == ".toml"
