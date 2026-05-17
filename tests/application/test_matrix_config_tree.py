from __future__ import annotations

import tomllib
import csv
from dataclasses import dataclass
from pathlib import Path

import pytest

from monocycle_nash.application.matrix_config_tree import (
    MatrixConfigTree,
    MatrixConfigTreeResolver,
    MatrixResolutionResult,
    ResolvedOutput,
)
from monocycle_nash.application.experiment_output_nodes import (
    TeamFeatureVectorCsvOutputNode,
    TeamFeatureVectorDirectedGraphOutputNode,
    TeamMatchupExperimentCsvOutputNode,
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
    def __init__(self, characters: tuple[Character, ...]):
        self._characters = characters

    def load_characters(self, path: str) -> tuple[Character, ...]:
        return self._characters


class StubTeamListFilePort(TeamListFilePort):
    def __init__(self, teams: tuple[Team, ...]):
        self._teams = teams

    def load_teams(self, path: str) -> tuple[Team, ...]:
        return self._teams


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
            characters=CharacterInlineSource(
                (
                    CharacterNode(power=1.0, vector=(1.0, 0.0), label="A"),
                    CharacterNode(power=0.0, vector=(0.0, 1.0), label="B"),
                    CharacterNode(power=-1.0, vector=(-1.0, 0.0), label="C"),
                ),
                name="characters",
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
        ("team-root", "characters"),
        ("team-root",),
    }


def test_resolver_builds_team_matrix_from_character_source_node(tmp_path: Path) -> None:
    tree = MatrixConfigTree(
        root=GeneralFromTeamMatchupsNode(
            name="team-root",
            teams=TeamInlineSource((
                TeamNode(label="A+B", member_ids=("A", "B")),
                TeamNode(label="B+C", member_ids=("B", "C")),
            )),
            characters=CharacterInlineSource(
                (
                    CharacterNode(power=1.0, vector=(1.0, 0.0), label="A"),
                    CharacterNode(power=0.0, vector=(0.0, 1.0), label="B"),
                    CharacterNode(power=-1.0, vector=(-1.0, 0.0), label="C"),
                ),
                name="characters",
                outputs=(CharacterVectorGraphOutputNode(filename="chars.svg"),),
            ),
            use_monocycle_formula=True,
        )
    )

    output_port = StubOutputPathPort(tmp_path)
    resolver = MatrixConfigTreeResolver(output_path_port=output_port)
    result = resolver.resolve(tree)

    assert isinstance(result.root.value, GeneralPayoffMatrix)
    assert result.root.value.matrix.shape == (2, 2)
    assert len(result.outputs) == 1
    assert output_port.calls[0][1] == ("team-root", "characters")


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


def test_resolver_runs_team_matchup_experiment_csv_output_node(tmp_path: Path) -> None:
    tree = MatrixConfigTree(
        root=GeneralFromTeamMatchupsNode(
            name="team-root",
            teams=TeamInlineSource((
                TeamNode(label="team_i", member_ids=("c1", "c2")),
                TeamNode(label="team_j1", member_ids=("c3", "c4")),
                TeamNode(label="team_j2", member_ids=("c4", "c5")),
            )),
            characters=CharacterInlineSource(
                (
                    CharacterNode(power=0.0, vector=(3.0, 0.0), label="c1"),
                    CharacterNode(power=0.0, vector=(0.0, 2.0), label="c2"),
                    CharacterNode(power=0.0, vector=(-1.0, 1.0), label="c3"),
                    CharacterNode(power=0.0, vector=(-1.0, -1.0), label="c4"),
                    CharacterNode(power=0.0, vector=(1.0, -1.0), label="c5"),
                ),
            ),
            use_monocycle_formula=False,
            outputs=(
                TeamMatchupExperimentCsvOutputNode(
                    filename="team_experiment.csv",
                    focus_team="team_i",
                ),
            ),
        )
    )

    output_port = StubOutputPathPort(tmp_path)
    resolver = MatrixConfigTreeResolver(output_path_port=output_port)
    result = resolver.resolve(tree)

    assert len(result.outputs) == 1
    output = result.outputs[0]
    assert output.path.exists()
    assert output_port.calls[0][2] == "team_matchup_experiment_csv"

    with output.path.open("r", encoding="utf-8", newline="") as f:
        records = list(csv.DictReader(f))

    assert len(records) == 2
    assert all(r["fixed_team_label"] == "team_i" for r in records)
    assert {r["j_team_label"] for r in records} == {"team_j1", "team_j2"}
    for record in records:
        assert record["j3_label"] in {"c3", "c4", "c5"}
        assert record["j4_label"] in {"c3", "c4", "c5"}
        angle_deg = float(record["angle_v12_to_v34_deg"])
        assert -180.0 <= angle_deg <= 180.0
        bij = float(record["bij"])
        j_index = int(record["j_team_index"])
        assert bij == pytest.approx(float(result.root.value.matrix[0, j_index]))


def test_resolver_runs_team_feature_vector_outputs(tmp_path: Path) -> None:
    tree = MatrixConfigTree(
        root=GeneralFromTeamMatchupsNode(
            name="team-root",
            teams=TeamInlineSource((
                TeamNode(label="team_i", member_ids=("c1", "c2")),
                TeamNode(label="team_j1", member_ids=("c3", "c4")),
            )),
            characters=CharacterInlineSource(
                (
                    CharacterNode(power=0.0, vector=(3.0, 0.0), label="c1"),
                    CharacterNode(power=0.0, vector=(0.0, 2.0), label="c2"),
                    CharacterNode(power=0.0, vector=(-1.0, 1.0), label="c3"),
                    CharacterNode(power=0.0, vector=(-1.0, -1.0), label="c4"),
                ),
            ),
            use_monocycle_formula=False,
            outputs=(
                TeamFeatureVectorCsvOutputNode(filename="team_feature_vectors.csv"),
                TeamFeatureVectorDirectedGraphOutputNode(filename="team_feature_vectors.svg"),
            ),
        )
    )
    output_port = StubOutputPathPort(tmp_path)
    resolver = MatrixConfigTreeResolver(output_path_port=output_port)
    result = resolver.resolve(tree)

    assert len(result.outputs) == 2
    csv_output = next(o for o in result.outputs if o.path.suffix == ".csv")
    svg_output = next(o for o in result.outputs if o.path.suffix == ".svg")
    assert csv_output.path.exists()
    assert svg_output.path.exists()
    assert "<svg" in svg_output.path.read_text(encoding="utf-8")

    with csv_output.path.open("r", encoding="utf-8", newline="") as f:
        records = list(csv.DictReader(f))
    assert len(records) == 2
    assert {r["team_label"] for r in records} == {"team_i", "team_j1"}
    for record in records:
        assert "feature_x" in record
        assert "feature_y" in record
        assert "feature_distance" in record
        assert "feature_angle_rad" in record
        assert "feature_angle_deg" in record
        assert float(record["feature_distance"]) >= 0.0


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
            outputs=(PayoffDirectedGraphOutputNode(filename="chars.svg"),),
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
            characters=CharacterInlineSource(
                (
                    CharacterNode(power=1.0, vector=(1.0, 0.0), label="A"),
                    CharacterNode(power=0.0, vector=(0.0, 1.0), label="B"),
                    CharacterNode(power=-1.0, vector=(-1.0, 0.0), label="C"),
                ),
                name="characters",
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


def test_resolver_allows_single_execute_for_multi_node_emissions(tmp_path: Path) -> None:
    execute_calls: list[tuple[str, ...]] = []

    @dataclass(frozen=True)
    class _CombinedTextOutputNode(OutputNode["ApplicationNode[str]"], output_method="combined_text"):
        runner: str | None = "final"
        filename: str = "combined.txt"

        @classmethod
        def _from_output_spec(cls, output_spec):  # pragma: no cover
            raise NotImplementedError

        def emit(
            self,
            *,
            node_name: str,
            node_path: tuple[str, ...],
            node: "ApplicationNode[str]",
        ) -> OutputEmission:
            return OutputEmission(
                output_node=self,
                node_name=node_name,
                node_path=node_path,
                runner=self.runner,
                node=node,
                payload=node.value,
            )

        def execute(
            self,
            *,
            output_path_port: OutputPathPort,
            run_id: str,
            node_path: tuple[str, ...],
            node: "ApplicationNode[str]",
            ctx: NodeResolutionContext,
        ) -> Path:  # pragma: no cover
            # このテストでは execute_emissions を使った集約実行のみを検証する。
            raise NotImplementedError

        def execute_emissions(
            self,
            *,
            output_path_port: OutputPathPort,
            run_id: str,
            emissions: tuple[OutputEmission, ...],
            ctx: NodeResolutionContext,
        ) -> tuple[Path, ...]:
            execute_calls.append(tuple(str(e.payload) for e in emissions))
            path = output_path_port.resolve_output_path(
                run_id=run_id,
                node_path=emissions[0].node_path,
                output_method=self.output_method,
                filename=self.filename,
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("\n".join(str(e.payload) for e in emissions), encoding="utf-8")
            return (path,)

    @dataclass
    class _DummyValueNode(ApplicationNode[str]):
        payload: str
        name: str
        outputs: tuple[OutputNode, ...] = ()

        def provide_object(
            self,
            *,
            ctx: NodeResolutionContext,
        ) -> str:
            return self.payload

    @dataclass
    class _BridgeMatrixNode(MatrixNode):
        left: _DummyValueNode
        right: _DummyValueNode
        name: str = "root"
        outputs: tuple[OutputNode, ...] = ()

        @classmethod
        def _from_spec(cls, matrix_spec, child_builder):  # pragma: no cover
            raise NotImplementedError

        def resolve_value(self, *, ctx: NodeResolutionContext):
            ctx.resolve_node(self.left)
            ctx.resolve_node(self.right)
            return GeneralPayoffMatrix([[0.0, 1.0], [-1.0, 0.0]], ["A", "B"])

    tree = MatrixConfigTree(
        root=_BridgeMatrixNode(
            left=_DummyValueNode(
                payload="left",
                name="left",
                outputs=(_CombinedTextOutputNode(runner="final"),),
            ),
            right=_DummyValueNode(
                payload="right",
                name="right",
                outputs=(_CombinedTextOutputNode(runner="final"),),
            ),
        )
    )
    output_port = StubOutputPathPort(tmp_path)
    resolver = MatrixConfigTreeResolver(output_path_port=output_port)

    result = resolver.resolve(tree)

    assert execute_calls == [("left", "right")]
    assert len(result.output_emissions) == 2
    assert len(result.runners) == 1
    assert result.runners[0].runner == "final"
    assert result.runners[0].emitted_count == 2
    assert result.runners[0].output_count == 1
    assert len(result.outputs) == 1
    assert result.outputs[0].path.read_text(encoding="utf-8") == "left\nright"


def test_execute_emissions_can_access_character_and_matrix_values(tmp_path: Path) -> None:
    @dataclass(frozen=True)
    class _UsecaseOutputNode(OutputNode["ApplicationNode[object]"], output_method="usecase_multi_domain"):
        runner: str | None = "usecase-a"
        filename: str = "usecase.txt"

        @classmethod
        def _from_output_spec(cls, output_spec):  # pragma: no cover
            raise NotImplementedError

        def emit(
            self,
            *,
            node_name: str,
            node_path: tuple[str, ...],
            node: "ApplicationNode[object]",
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
            node: "ApplicationNode[object]",
            ctx: NodeResolutionContext,
        ) -> Path:  # pragma: no cover
            raise NotImplementedError

        def execute_emissions(
            self,
            *,
            output_path_port: OutputPathPort,
            run_id: str,
            emissions: tuple[OutputEmission, ...],
            ctx: NodeResolutionContext,
        ) -> tuple[Path, ...]:
            character_count = 0
            matrix_shape: tuple[int, int] | None = None
            for emission in emissions:
                value = emission.resolve_value(ctx=ctx)
                if isinstance(value, tuple) and value and isinstance(value[0], Character):
                    character_count = len(value)
                if isinstance(value, GeneralPayoffMatrix):
                    matrix_shape = value.matrix.shape
            path = output_path_port.resolve_output_path(
                run_id=run_id,
                node_path=emissions[0].node_path,
                output_method=self.output_method,
                filename=self.filename,
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                f"characters={character_count},matrix_shape={matrix_shape}",
                encoding="utf-8",
            )
            return (path,)

    @dataclass
    class _CharacterValueNode(ApplicationNode[tuple[Character, ...]]):
        name: str = "characters"
        outputs: tuple[OutputNode, ...] = ()

        def provide_object(
            self,
            *,
            ctx: NodeResolutionContext,
        ) -> tuple[Character, ...]:
            return (
                Character(1.0, MatchupVector(1.0, 0.0), "A"),
                Character(0.0, MatchupVector(0.0, 1.0), "B"),
            )

    @dataclass
    class _MatrixValueNode(ApplicationNode[GeneralPayoffMatrix]):
        name: str = "matrix"
        outputs: tuple[OutputNode, ...] = ()

        def provide_object(
            self,
            *,
            ctx: NodeResolutionContext,
        ) -> GeneralPayoffMatrix:
            return GeneralPayoffMatrix([[0.0, 1.0], [-1.0, 0.0]], ["A", "B"])

    @dataclass
    class _BridgeMatrixNode(MatrixNode):
        characters_node: _CharacterValueNode
        matrix_node: _MatrixValueNode
        name: str = "root"
        outputs: tuple[OutputNode, ...] = ()

        @classmethod
        def _from_spec(cls, matrix_spec, child_builder):  # pragma: no cover
            raise NotImplementedError

        def resolve_value(self, *, ctx: NodeResolutionContext):
            ctx.resolve_node(self.characters_node)
            resolved_matrix = ctx.get_node_value(self.matrix_node)
            return resolved_matrix

    marker = _UsecaseOutputNode(runner="usecase-a")
    tree = MatrixConfigTree(
        root=_BridgeMatrixNode(
            characters_node=_CharacterValueNode(outputs=(marker,)),
            matrix_node=_MatrixValueNode(outputs=(marker,)),
        )
    )
    output_port = StubOutputPathPort(tmp_path)
    resolver = MatrixConfigTreeResolver(output_path_port=output_port)

    result = resolver.resolve(tree)

    assert len(result.output_emissions) == 2
    assert len(result.runners) == 1
    assert result.runners[0].runner == "usecase-a"
    assert result.runners[0].output_count == 1
    assert len(result.outputs) == 1
    assert result.outputs[0].path.read_text(encoding="utf-8") == "characters=2,matrix_shape=(2, 2)"


def test_monocycle_node_resolves_characters_from_character_source() -> None:
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
            if isinstance(node, ApplicationNode):
                node.set_value(node.provide_object(ctx=self))
                return node
            raise NotImplementedError

        def get_node_value(self, node: object) -> object:
            if node is target_node:
                return resolved
            if isinstance(node, ApplicationNode):
                return self.resolve_node(node).value
            raise NotImplementedError

    characters = target_node.characters.provide_object(ctx=_Ctx())
    assert len(characters) == 2
    assert [c.label for c in characters] == ["A", "B"]


def test_character_vector_output_executes_for_character_source(tmp_path: Path) -> None:
    source = CharacterInlineSource((
        CharacterNode(power=1.0, vector=(1.0, 0.0), label="A"),
        CharacterNode(power=0.0, vector=(0.0, 1.0), label="B"),
    ))
    output_node = CharacterVectorGraphOutputNode(filename="team_chars.svg")

    class _Ctx(NodeResolutionContext):
        def resolve_node(self, node: object) -> object:
            if isinstance(node, ApplicationNode):
                node.set_value(node.provide_object(ctx=self))
                return node
            raise NotImplementedError

        def get_node_value(self, node: object) -> object:
            if isinstance(node, ApplicationNode):
                return self.resolve_node(node).value
            raise NotImplementedError

    path = output_node.execute(
        output_path_port=StubOutputPathPort(tmp_path),
        run_id="1",
        node_path=("source",),
        node=source,
        ctx=_Ctx(),
    )

    assert path.exists()


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

    @dataclass
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

    @dataclass
    class _BridgeMatrixNode(MatrixNode):
        child: _DummyValueNode
        name: str = "root"
        outputs: tuple[OutputNode, ...] = ()

        @classmethod
        def _from_spec(cls, spec, build_child):  # pragma: no cover
            raise NotImplementedError

        def resolve_value(self, *, ctx: NodeResolutionContext):
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


def test_resolver_resolves_shared_node_once(
    tmp_path: Path,
) -> None:
    call_count = [0]

    @dataclass(frozen=True)
    class _DummyTextOutputNode(OutputNode["_CountingValueNode"], output_method="dummy_text_counting"):
        runner: str | None = None
        filename: str = "leaf.txt"

        @classmethod
        def _from_output_spec(cls, output_spec):  # pragma: no cover
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

    @dataclass
    class _CountingValueNode(ApplicationNode[str]):
        name: str = "leaf"
        outputs: tuple[OutputNode, ...] = ()

        def provide_object(
            self,
            *,
            ctx: NodeResolutionContext,
        ) -> str:
            call_count[0] += 1
            return f"value-{call_count[0]}"

    @dataclass
    class _BridgeMatrixNode(MatrixNode):
        child: _CountingValueNode
        name: str = "root"
        outputs: tuple[OutputNode, ...] = ()

        @classmethod
        def _from_spec(cls, matrix_spec, child_builder):  # pragma: no cover
            raise NotImplementedError

        def resolve_value(self, *, ctx: NodeResolutionContext):
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

    result = resolver.resolve(tree)

    assert call_count[0] == 1
    assert len(result.output_emissions) == 1
    assert len(result.outputs) == 1
    assert [run_id for run_id, _, _, _ in output_port.calls] == [str(result.run_id)]


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
    characters = (
        Character(1.0, MatchupVector(1.0, 0.0), "A"),
        Character(0.0, MatchupVector(0.0, 1.0), "B"),
    )
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
    teams = (
        Team(label="A+B", member_ids=("A", "B")),
        Team(label="B+C", member_ids=("B", "C")),
    )
    tree = MatrixConfigTree(
        root=GeneralFromTeamMatchupsNode(
            teams=TeamListFromFileNode(path="teams.toml"),
            characters=characters,
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
            characters=CharacterInlineSource((
                CharacterNode(power=1.0, vector=(1.0, 0.0), label="A"),
                CharacterNode(power=0.0, vector=(0.0, 1.0), label="B"),
            )),
        )
    )
    resolver = MatrixConfigTreeResolver()

    with pytest.raises(ValueError, match="TeamListFilePort"):
        resolver.resolve(tree)
