from __future__ import annotations

from pathlib import Path

import pytest

from monocycle_nash.application.matrix_config_tree import (
    MatrixConfigTree,
    MatrixConfigTreeResolver,
    MatrixResolutionResult,
    ResolvedOutput,
)
from monocycle_nash.application.matrix_nodes import (
    ApproxMonocycleToGeneralNode,
    CharacterInlineSource,
    CharacterListFromFileNode,
    CharacterNode,
    CharacterVectorGraphOutputNode,
    GeneralFromTeamMatchupsNode,
    MonocycleFromCharactersNode,
    PayoffDirectedGraphOutputNode,
    TeamInlineSource,
    TeamListFromFileNode,
    TeamNode,
)
from monocycle_nash.application.ports import (
    CharacterListFilePort,
    OutputPathPort,
    TeamListFilePort,
)
from monocycle_nash.domain.character import Character, MatchupVector
from monocycle_nash.domain.matrix.general import GeneralPayoffMatrix
from monocycle_nash.domain.matrix.monocycle import MonocyclePayoffMatrix
from monocycle_nash.domain.team import Team


class StubOutputPathPort(OutputPathPort):
    def __init__(self, base_dir: Path):
        self._base_dir = base_dir

    def resolve_output_path(
        self,
        *,
        node_name: str,
        output_method: str,
        filename: str,
    ) -> Path:
        return self._base_dir / f"{node_name}-{output_method}-{filename}"


class StubCharacterListFilePort(CharacterListFilePort):
    def __init__(self, characters: list[Character]):
        self._characters = characters

    def load_characters(self, path: str) -> list[Character]:
        return list(self._characters)


class StubTeamListFilePort(TeamListFilePort):
    def __init__(self, teams: list[Team]):
        self._teams = teams

    def load_teams(self, path: str) -> list[Team]:
        return list(self._teams)


# ---------------------------------------------------------------------------
# Typed node construction
# ---------------------------------------------------------------------------


def test_approx_node_wraps_monocycle_node() -> None:
    """ApproxMonocycleToGeneralNode が MonocycleFromCharactersNode を source として持てる。"""
    source_node = MonocycleFromCharactersNode(
        characters=CharacterInlineSource((
            CharacterNode(power=1.0, vector=(1.0, 0.0), label="A"),
            CharacterNode(power=0.5, vector=(0.0, 1.0), label="B"),
        ))
    )
    approx_node = ApproxMonocycleToGeneralNode(
        source=source_node,
        outputs=(PayoffDirectedGraphOutputNode(filename="root.svg", threshold=0.2),),
    )
    tree = MatrixConfigTree(root=approx_node)

    assert tree.root is approx_node
    assert tree.root.source is source_node  # type: ignore[union-attr]
    assert tree.root.outputs[0].filename == "root.svg"  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# Resolver — nested dependency with outputs
# ---------------------------------------------------------------------------


def test_resolver_builds_team_matrix_with_nested_dependency_and_outputs(tmp_path: Path) -> None:
    tree = MatrixConfigTree(
        root=GeneralFromTeamMatchupsNode(
            name="team-root",
            teams=TeamInlineSource((
                TeamNode(label="A+B", member_ids=("A", "B")),
                TeamNode(label="B+C", member_ids=("B", "C")),
            )),
            character_matrix=MonocycleFromCharactersNode(
                name="character-source",
                characters=CharacterInlineSource((
                    CharacterNode(power=1.0, vector=(1.0, 0.0), label="A"),
                    CharacterNode(power=0.0, vector=(0.0, 1.0), label="B"),
                    CharacterNode(power=-1.0, vector=(-1.0, 0.0), label="C"),
                )),
                labels=["A", "B", "C"],
                outputs=(CharacterVectorGraphOutputNode(filename="chars.svg"),),
            ),
            use_monocycle_formula=True,
            outputs=(PayoffDirectedGraphOutputNode(filename="team.svg"),),
        )
    )

    resolver = MatrixConfigTreeResolver(output_path_port=StubOutputPathPort(tmp_path))
    result = resolver.resolve(tree)

    assert isinstance(result.root, GeneralPayoffMatrix)
    assert result.root.matrix.shape == (2, 2)
    assert len(result.outputs) == 2
    assert all(output.path.exists() for output in result.outputs)


# ---------------------------------------------------------------------------
# Resolver — without output port rejects output execution
# ---------------------------------------------------------------------------


def test_resolver_without_output_path_port_rejects_output() -> None:
    tree = MatrixConfigTree(
        root=MonocycleFromCharactersNode(
            characters=CharacterInlineSource((
                CharacterNode(power=1.0, vector=(1.0, 0.0), label="A"),
                CharacterNode(power=0.0, vector=(0.0, 1.0), label="B"),
            )),
            outputs=(CharacterVectorGraphOutputNode(filename="chars.svg"),),
        )
    )
    resolver = MatrixConfigTreeResolver()

    with pytest.raises(ValueError, match="OutputPathPort"):
        resolver.resolve(tree)


# ---------------------------------------------------------------------------
# Resolver — inline character nodes produce MonocyclePayoffMatrix
# ---------------------------------------------------------------------------


def test_resolver_monocycle_node_produces_monocycle_matrix() -> None:
    tree = MatrixConfigTree(
        root=MonocycleFromCharactersNode(
            characters=CharacterInlineSource((
                CharacterNode(power=1.0, vector=(1.0, 0.0), label="A"),
                CharacterNode(power=0.2, vector=(0.0, 1.0), label="B"),
            )),
            labels=["A", "B"],
        )
    )

    result = MatrixConfigTreeResolver().resolve(tree)
    assert isinstance(result.root, MonocyclePayoffMatrix)
    assert result.root.labels == ["A", "B"]


# ---------------------------------------------------------------------------
# Resolver — file-backed character loading via CharacterListFilePort
# ---------------------------------------------------------------------------


def test_resolver_file_backed_characters(tmp_path: Path) -> None:
    characters = [
        Character(1.0, MatchupVector(1.0, 0.0), "A"),
        Character(0.0, MatchupVector(0.0, 1.0), "B"),
    ]
    tree = MatrixConfigTree(
        root=MonocycleFromCharactersNode(
            characters=CharacterListFromFileNode(path="dummy.toml"),
        )
    )

    resolver = MatrixConfigTreeResolver(
        character_list_file_port=StubCharacterListFilePort(characters)
    )
    result = resolver.resolve(tree)
    assert isinstance(result.root, MonocyclePayoffMatrix)
    assert result.root.matrix.shape == (2, 2)


def test_resolver_file_backed_characters_without_port_raises() -> None:
    tree = MatrixConfigTree(
        root=MonocycleFromCharactersNode(
            characters=CharacterListFromFileNode(path="dummy.toml"),
        )
    )
    resolver = MatrixConfigTreeResolver()

    with pytest.raises(ValueError, match="CharacterListFilePort"):
        resolver.resolve(tree)


# ---------------------------------------------------------------------------
# Resolver — file-backed team loading via TeamListFilePort
# ---------------------------------------------------------------------------


def test_resolver_file_backed_teams(tmp_path: Path) -> None:
    character_source = CharacterInlineSource((
        CharacterNode(power=1.0, vector=(1.0, 0.0), label="A"),
        CharacterNode(power=0.0, vector=(0.0, 1.0), label="B"),
        CharacterNode(power=-1.0, vector=(-1.0, 0.0), label="C"),
    ))
    teams = [
        Team(label="A+B", member_ids=("A", "B")),
        Team(label="B+C", member_ids=("B", "C")),
    ]
    tree = MatrixConfigTree(
        root=GeneralFromTeamMatchupsNode(
            teams=TeamListFromFileNode(path="teams.toml"),
            character_matrix=MonocycleFromCharactersNode(
                characters=character_source,
                labels=["A", "B", "C"],
            ),
        )
    )

    resolver = MatrixConfigTreeResolver(
        team_list_file_port=StubTeamListFilePort(teams)
    )
    result = resolver.resolve(tree)
    assert isinstance(result.root, GeneralPayoffMatrix)
    assert result.root.matrix.shape == (2, 2)


def test_resolver_file_backed_teams_without_port_raises() -> None:
    tree = MatrixConfigTree(
        root=GeneralFromTeamMatchupsNode(
            teams=TeamListFromFileNode(path="teams.toml"),
            character_matrix=MonocycleFromCharactersNode(
                characters=CharacterInlineSource((
                    CharacterNode(power=1.0, vector=(1.0, 0.0), label="A"),
                    CharacterNode(power=0.0, vector=(0.0, 1.0), label="B"),
                )),
            ),
        )
    )
    resolver = MatrixConfigTreeResolver()

    with pytest.raises(ValueError, match="TeamListFilePort"):
        resolver.resolve(tree)

