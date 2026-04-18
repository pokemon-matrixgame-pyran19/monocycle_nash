from __future__ import annotations

import dataclasses

import pytest

from monocycle_nash.application.node_spec import NodeSpec, OutputSpec
from monocycle_nash.application.snapshot import ConfigTreeSnapshot


def test_snapshot_holds_root_node_spec() -> None:
    """ConfigTreeSnapshot がルート NodeSpec を保持できる。"""
    spec = NodeSpec(method="general_from_raw", name="root")
    snapshot = ConfigTreeSnapshot(root=spec)
    assert snapshot.root is spec


def test_snapshot_is_frozen() -> None:
    """ConfigTreeSnapshot は frozen dataclass であり再代入できない。"""
    spec = NodeSpec(method="general_from_raw")
    snapshot = ConfigTreeSnapshot(root=spec)
    with pytest.raises(dataclasses.FrozenInstanceError):
        snapshot.root = NodeSpec(method="other")  # type: ignore[misc]


def test_snapshot_equality() -> None:
    """同じ NodeSpec から作った ConfigTreeSnapshot は等値になる。"""
    spec = NodeSpec(
        method="monocycle_from_characters",
        name="rps",
        params={"key": "value"},
        outputs=(OutputSpec(method="payoff_directed_graph"),),
    )
    s1 = ConfigTreeSnapshot(root=spec)
    s2 = ConfigTreeSnapshot(root=spec)
    assert s1 == s2
