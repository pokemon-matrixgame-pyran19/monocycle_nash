from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

import pytest

from monocycle_nash.application.matrix_config_tree import (
    MatrixConfigTree,
    MatrixConfigTreeResolver,
    MatrixResolutionResult,
    ResolvedOutput,
)
from monocycle_nash.application.matrix_nodes import (
    ApplicationNode,
    ApproxMonocycleToGeneralNode,
    CharacterInlineSource,
    CharacterListFromFileNode,
    CharacterNode,
    CharacterVectorGraphOutputNode,
    EquilibriumOutputNode,
    GeneralFromTeamMatchupsNode,
    MatrixNode,
    MonocycleFromCharactersNode,
    NodeResolutionContext,
    OutputEmission,
    OutputNode,
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
        self.calls: list[tuple[str, tuple[str, ...], str, str]] = []

    def resolve_output_path(
        self,
        *,
        run_id: str,
        node_path: tuple[str, ...],
        output_method: str,
        filename: str,
    ) -> Path:
        self.calls.append((run_id, node_path, output_method, filename))
        return self._base_dir / run_id / Path(*node_path) / output_method / filename


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

    output_port = StubOutputPathPort(tmp_path)
    resolver = MatrixConfigTreeResolver(output_path_port=output_port)
    result = resolver.resolve(tree)

    assert isinstance(result.root.value, GeneralPayoffMatrix)
    assert result.root.value.matrix.shape == (2, 2)
    assert result.run_id == 1
    assert len(result.outputs) == 2
    assert len(result.output_emissions) == 2
    assert all(output.path.exists() for output in result.outputs)
    assert len(result.runners) == 2
    assert all(
        run_id == str(result.run_id)
        for run_id, _, _, _ in output_port.calls
    )
    assert set(node_path for _, node_path, _, _ in output_port.calls) == {
        ("team-root", "character-source"),
        ("team-root",),
    }


def test_resolver_runs_equilibrium_output_node(tmp_path: Path) -> None:
    tree = MatrixConfigTree(
        root=MonocycleFromCharactersNode(
            name="mono",
            characters=CharacterInlineSource((
                CharacterNode(power=1.0, vector=(1.0, 0.0), label="A"),
                CharacterNode(power=0.0, vector=(0.0, 1.0), label="B"),
                CharacterNode(power=-1.0, vector=(-1.0, 0.0), label="C"),
            )),
            outputs=(EquilibriumOutputNode(filename="eq.toml"),),
        )
    )

    output_port = StubOutputPathPort(tmp_path)
    resolver = MatrixConfigTreeResolver(output_path_port=output_port)
    result = resolver.resolve(tree)

    assert len(result.outputs) == 1
    assert len(result.runners) == 1
    output = result.outputs[0]
    assert output.path.exists()
    assert output_port.calls[0][2] == "equilibrium"
    data = tomllib.loads(output.path.read_text(encoding="utf-8"))
    assert "strategies" in data
    assert len(data["strategies"]) == 3
    probabilities = [float(s["probability"]) for s in data["strategies"]]
    assert pytest.approx(sum(probabilities), rel=1e-6, abs=1e-6) == 1.0


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


def test_resolver_runs_shared_runner_once_after_full_resolution(tmp_path: Path) -> None:
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
                outputs=(CharacterVectorGraphOutputNode(runner="final", filename="chars.svg"),),
            ),
            use_monocycle_formula=True,
            outputs=(PayoffDirectedGraphOutputNode(runner="final", filename="team.svg"),),
        )
    )
    output_port = StubOutputPathPort(tmp_path)
    resolver = MatrixConfigTreeResolver(output_path_port=output_port)

    result = resolver.resolve(tree)

    assert len(result.output_emissions) == 2
    assert len(result.runners) == 1
    assert result.runners[0].runner == "final"
    assert result.runners[0].emitted_count == 2
    assert result.runners[0].output_count == 2
    assert all(o.runner == "final" for o in result.outputs)


def test_monocycle_node_provides_characters_without_matrix_property() -> None:
    target_node = MonocycleFromCharactersNode(
        characters=CharacterInlineSource((
            CharacterNode(power=1.0, vector=(1.0, 0.0), label="A"),
            CharacterNode(power=0.0, vector=(0.0, 1.0), label="B"),
        )),
    )
    resolved = GeneralPayoffMatrix([[0.0, 1.0], [-1.0, 0.0]], ["A", "B"])
    # 直接 context 実装を使って source から characters を供給できることを確認する
    class _Ctx(NodeResolutionContext):
        def resolve_node(self, node: object) -> object:
            if node is target_node:
                target_node.set_value(resolved)
                return target_node
            raise NotImplementedError

        def get_node_value(self, node: object) -> object:
            if node is target_node:
                return resolved
            raise NotImplementedError

        def load_characters_from_file(self, path: str) -> list[Character]:
            raise NotImplementedError

        def load_teams_from_file(self, path: str) -> list[Team]:
            raise NotImplementedError

    characters = target_node.provide_characters(ctx=_Ctx())
    assert len(characters) == 2
    assert [c.label for c in characters] == ["A", "B"]


def test_team_matchups_node_can_emit_character_vector_from_child_domain(tmp_path: Path) -> None:
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
            ),
            outputs=(CharacterVectorGraphOutputNode(filename="team_chars.svg"),),
        )
    )
    output_port = StubOutputPathPort(tmp_path)
    resolver = MatrixConfigTreeResolver(output_path_port=output_port)

    result = resolver.resolve(tree)

    assert len(result.outputs) == 1
    assert result.outputs[0].path.exists()


def test_resolver_resolves_non_matrix_node_with_name_and_outputs(tmp_path: Path) -> None:
    @dataclass(frozen=True)
    class _DummyTextOutputNode(OutputNode["_DummyValueNode"], output_method="dummy_text"):
        runner: str | None = None
        filename: str = "dummy.txt"

        @classmethod
        def _from_output_spec(cls, spec):  # pragma: no cover
            raise NotImplementedError

        def emit(
            self,
            *,
            node_name: str,
            node_path: tuple[str, ...],
            node: "_DummyValueNode",
        ) -> OutputEmission:
            return OutputEmission(
                output_node=self,
                node_name=node_name,
                node_path=node_path,
                runner=self.runner,
                node=node,
            )

        def execute(
            self,
            *,
            output_path_port: OutputPathPort,
            run_id: str,
            node_path: tuple[str, ...],
            node: "_DummyValueNode",
            ctx: NodeResolutionContext,
        ) -> Path:
            path = output_path_port.resolve_output_path(
                run_id=run_id,
                node_path=node_path,
                output_method=self.output_method,
                filename=self.filename,
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(node.provide_object(ctx=ctx), encoding="utf-8")
            return path

    @dataclass(frozen=True)
    class _DummyValueNode(ApplicationNode[str]):
        payload: str
        name: str = "payload"
        outputs: tuple[OutputNode, ...] = ()

        def provide_object(
            self,
            *,
            ctx: NodeResolutionContext,
        ) -> str:
            return self.payload

    @dataclass(frozen=True)
    class _BridgeMatrixNode(MatrixNode):
        child: _DummyValueNode
        name: str = "root"
        outputs: tuple[OutputNode, ...] = ()

        @classmethod
        def _from_spec(cls, spec, build_child):  # pragma: no cover
            raise NotImplementedError

        def build(self, ctx: NodeResolutionContext):
            ctx.resolve_node(self.child)
            return GeneralPayoffMatrix([[0.0, 1.0], [-1.0, 0.0]], ["A", "B"])

    tree = MatrixConfigTree(
        root=_BridgeMatrixNode(
            child=_DummyValueNode(
                payload="hello",
                outputs=(_DummyTextOutputNode(filename="dummy.txt"),),
            ),
        )
    )
    output_port = StubOutputPathPort(tmp_path)
    resolver = MatrixConfigTreeResolver(output_path_port=output_port)

    result = resolver.resolve(tree)

    assert len(result.outputs) == 1
    assert result.outputs[0].node_name == "payload"
    assert output_port.calls[0][1] == ("root", "payload")
    assert result.outputs[0].path.read_text(encoding="utf-8") == "hello"


def test_resolver_resolves_shared_node_once_per_run_and_re_resolves_next_run(
    tmp_path: Path,
) -> None:
    call_count = {"count": 0}

    @dataclass(frozen=True)
    class _DummyTextOutputNode(OutputNode["_CountingValueNode"], output_method="dummy_text_counting"):
        runner: str | None = None
        filename: str = "leaf.txt"

        @classmethod
        def _from_output_spec(cls, spec):  # pragma: no cover
            raise NotImplementedError

        def emit(
            self,
            *,
            node_name: str,
            node_path: tuple[str, ...],
            node: "_CountingValueNode",
        ) -> OutputEmission:
            return OutputEmission(
                output_node=self,
                node_name=node_name,
                node_path=node_path,
                runner=self.runner,
                node=node,
            )

        def execute(
            self,
            *,
            output_path_port: OutputPathPort,
            run_id: str,
            node_path: tuple[str, ...],
            node: "_CountingValueNode",
            ctx: NodeResolutionContext,
        ) -> Path:
            path = output_path_port.resolve_output_path(
                run_id=run_id,
                node_path=node_path,
                output_method=self.output_method,
                filename=self.filename,
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(node.value, encoding="utf-8")
            return path

    @dataclass(frozen=True)
    class _CountingValueNode(ApplicationNode[str]):
        name: str = "leaf"
        outputs: tuple[OutputNode, ...] = ()

        def provide_object(
            self,
            *,
            ctx: NodeResolutionContext,
        ) -> str:
            call_count["count"] += 1
            return f"value-{call_count['count']}"

    @dataclass(frozen=True)
    class _BridgeMatrixNode(MatrixNode):
        child: _CountingValueNode
        name: str = "root"
        outputs: tuple[OutputNode, ...] = ()

        @classmethod
        def _from_spec(cls, spec, build_child):  # pragma: no cover
            raise NotImplementedError

        def build(self, ctx: NodeResolutionContext):
            ctx.resolve_node(self.child)
            ctx.resolve_node(self.child)
            return GeneralPayoffMatrix([[0.0, 1.0], [-1.0, 0.0]], ["A", "B"])

    tree = MatrixConfigTree(
        root=_BridgeMatrixNode(
            child=_CountingValueNode(
                outputs=(_DummyTextOutputNode(filename="leaf.txt"),),
            ),
        )
    )
    output_port = StubOutputPathPort(tmp_path)
    resolver = MatrixConfigTreeResolver(output_path_port=output_port)

    first = resolver.resolve(tree)
    second = resolver.resolve(tree)

    assert call_count["count"] == 2
    assert second.run_id > first.run_id
    assert len(first.output_emissions) == 1
    assert len(first.outputs) == 1
    assert len(second.output_emissions) == 1
    assert len(second.outputs) == 1
    assert [run_id for run_id, _, _, _ in output_port.calls] == [
        str(first.run_id),
        str(second.run_id),
    ]


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
    assert isinstance(result.root.value, MonocyclePayoffMatrix)
    assert result.root.value.labels == ["A", "B"]


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
    assert isinstance(result.root.value, MonocyclePayoffMatrix)
    assert result.root.value.matrix.shape == (2, 2)


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
    characters = CharacterInlineSource((
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
                characters=characters,
                labels=["A", "B", "C"],
            ),
        )
    )

    resolver = MatrixConfigTreeResolver(
        team_list_file_port=StubTeamListFilePort(teams)
    )
    result = resolver.resolve(tree)
    assert isinstance(result.root.value, GeneralPayoffMatrix)
    assert result.root.value.matrix.shape == (2, 2)


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
