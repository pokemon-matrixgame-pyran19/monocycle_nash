from __future__ import annotations

from pathlib import Path

from monocycle_nash.infrastructure.output import FileSystemOutputPathPort


def test_resolve_output_path_uses_run_id_and_node_hierarchy(tmp_path: Path) -> None:
    port = FileSystemOutputPathPort(result_base_dir=tmp_path / "result")

    path = port.resolve_output_path(
        run_id="run-001",
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
        run_id="run/../001",
        node_path=("team root", "../child"),
        output_method="payoff/directed",
        filename="../../unsafe?.svg",
    )

    assert path == (
        tmp_path
        / "result"
        / "run_001"
        / "team_root"
        / "child"
        / "payoff_directed"
        / "unsafe.svg"
    )


def test_resolve_output_path_distinguishes_empty_and_dot_only_component(
    tmp_path: Path,
) -> None:
    port = FileSystemOutputPathPort(result_base_dir=tmp_path / "result")

    empty_path = port.resolve_output_path(
        run_id="",
        node_path=("root",),
        output_method="graph",
        filename="a.svg",
    )
    dot_path = port.resolve_output_path(
        run_id="...",
        node_path=("root",),
        output_method="graph",
        filename="b.svg",
    )

    assert empty_path.relative_to(tmp_path / "result").parts[0] == "_empty"
    assert dot_path.relative_to(tmp_path / "result").parts[0] == "_dot"
