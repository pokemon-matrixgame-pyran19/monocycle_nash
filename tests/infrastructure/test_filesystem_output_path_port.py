from __future__ import annotations

from pathlib import Path

from monocycle_nash.infrastructure.output import FileSystemOutputPathPort


def test_resolve_output_path_uses_execution_unit_and_node_hierarchy(tmp_path: Path) -> None:
    port = FileSystemOutputPathPort(result_base_dir=tmp_path / "result")

    path = port.resolve_output_path(
        execution_unit_id="run-001",
        node_path=("team-root", "character-source"),
        output_method="character_vector_graph",
        filename="chars.svg",
    )

    assert path == (
        tmp_path
        / "result"
        / "run-001"
        / "team-root"
        / "character-source"
        / "character_vector_graph"
        / "chars.svg"
    )
    assert path.parent.exists()


def test_resolve_output_path_sanitizes_path_components(tmp_path: Path) -> None:
    port = FileSystemOutputPathPort(result_base_dir=tmp_path / "result")

    path = port.resolve_output_path(
        execution_unit_id="run/../001",
        node_path=("team root", "../child"),
        output_method="payoff/directed",
        filename="../../unsafe?.svg",
    )

    assert path == (
        tmp_path
        / "result"
        / "run____001"
        / "team_root"
        / "child"
        / "payoff_directed"
        / "unsafe_.svg"
    )
