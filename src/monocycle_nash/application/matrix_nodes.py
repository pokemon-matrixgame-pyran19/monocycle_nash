"""Application-layer typed node definitions for payoff matrix construction.

各ノード型が「この生成方式はこういう値や設定を受け取る」を明示する。
依存する他ドメインモデルは型付きフィールドとして直接保持し、
各ノードは build / emit / execute / load_characters / load_teams でそれぞれの解決ロジックを担う。

新規ノード種別を追加する場合は:
  1. 具象 MatrixNode サブクラスを作り、class 宣言に `node_method="..."` を付ける
  2. `_from_spec(cls, spec, build_child)` classmethod を実装する
  以上のみ。MatrixNodeFactory や登録処理への変更は不要。

新規出力種別を追加する場合は:
  1. 具象 OutputNode サブクラスを作り、class 宣言に `output_method="..."` を付ける
  2. `_from_output_spec(cls, spec)` classmethod を実装する
  3. `emit(...)` と `execute(...)` を実装する
"""

from __future__ import annotations

import numpy as np
import tomli_w
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, ClassVar, Generic, TypeVar

from monocycle_nash.application.node_spec import NodeSpec, OutputSpec
from monocycle_nash.application.ports import OutputPathPort
from monocycle_nash.domain.character import Character, MatchupVector
from monocycle_nash.domain.matrix.approximation import (
    DominantEigenpairMonocycleApproximation,
    EquilibriumPreservingResidualMonocycleApproximation,
    MonocycleToGeneralApproximation,
)
from monocycle_nash.domain.matrix.base import PayoffMatrix
from monocycle_nash.domain.matrix.builder import PayoffMatrixBuilder
from monocycle_nash.domain.solver.selector import SolverSelector
from monocycle_nash.domain.team import Team
from monocycle_nash.domain.visualization.character_vector_graph import CharacterVectorGraphPlotter
from monocycle_nash.domain.visualization.payoff_graph import PayoffDirectedGraphPlotter


# ---------------------------------------------------------------------------
# NodeDomainObject — ノードが提供するドメインオブジェクト型
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MatrixDomainObject:
    matrix: PayoffMatrix


@dataclass(frozen=True)
class CharactersDomainObject:
    characters: tuple[Character, ...]


@dataclass(frozen=True)
class TeamsDomainObject:
    teams: tuple[Team, ...]


@dataclass(frozen=True)
class MatrixCharactersDomainObject:
    matrix: PayoffMatrix
    characters: tuple[Character, ...]


@dataclass(frozen=True)
class MatrixTeamsDomainObject:
    matrix: PayoffMatrix
    teams: tuple[Team, ...]


@dataclass(frozen=True)
class MatrixCharactersTeamsDomainObject:
    matrix: PayoffMatrix
    characters: tuple[Character, ...]
    teams: tuple[Team, ...]


NodeDomainObject = (
    MatrixDomainObject
    | CharactersDomainObject
    | TeamsDomainObject
    | MatrixCharactersDomainObject
    | MatrixTeamsDomainObject
    | MatrixCharactersTeamsDomainObject
)

DomainT = TypeVar("DomainT", bound=NodeDomainObject)


# ---------------------------------------------------------------------------
# NodeResolutionContext — ノードが使用するコンテキストインターフェース
# ---------------------------------------------------------------------------


class NodeResolutionContext(ABC):
    """ApplicationNode 解決時に再帰参照やファイル読み込みで使うコンテキスト。"""

    @abstractmethod
    def resolve_node(self, node: MatrixNode) -> PayoffMatrix:
        """別ノードを再帰的に解決して PayoffMatrix を返す。"""
        raise NotImplementedError

    @abstractmethod
    def resolve_node_domains(self, node: "ApplicationNode[DomainT]") -> DomainT:
        """別ノードの解決済みドメインオブジェクトを返す。"""
        raise NotImplementedError

    @abstractmethod
    def load_characters_from_file(self, path: str) -> list[Character]:
        """ファイルからキャラクターリストを読み込む。"""
        raise NotImplementedError

    @abstractmethod
    def load_teams_from_file(self, path: str) -> list[Team]:
        """ファイルからチームリストを読み込む。"""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# ApplicationNode — 全ノード共通の抽象基底
# ---------------------------------------------------------------------------


class ApplicationNode(ABC, Generic[DomainT]):
    """解決済みドメインオブジェクトを提供する全ノード共通抽象。"""

    @abstractmethod
    def provide_domains(
        self,
        *,
        ctx: NodeResolutionContext,
        resolved: PayoffMatrix | None = None,
    ) -> DomainT:
        """ノードが提供するドメインオブジェクトを返す。

        resolved は行列構築ノードのように build 結果を持つノードで利用し、
        CharacterSource / TeamSource のような補助ノードでは未使用でよい。
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Character nodes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CharacterNode:
    """インラインのキャラクター設定ノード。"""

    power: float
    vector: tuple[float, float]
    label: str = ""


class CharacterSource(ApplicationNode[CharactersDomainObject]):
    """キャラクター入力ソースの抽象基底。"""

    @classmethod
    def from_node_spec(cls, spec: NodeSpec) -> CharacterSource:
        """NodeSpec の refs または params.characters からキャラクターソースを生成する。"""
        if "characters" in spec.refs:
            return CharacterListFromFileNode(path=spec.refs["characters"])
        chars_data = spec.params.get("characters", [])
        return CharacterInlineSource(
            characters=tuple(
                CharacterNode(
                    power=c["power"],
                    vector=(float(c["vector"][0]), float(c["vector"][1])),
                    label=c.get("label", ""),
                )
                for c in chars_data
            )
        )

    @abstractmethod
    def load_characters(self, ctx: NodeResolutionContext) -> list[Character]:
        """キャラクターリストを返す。"""
        raise NotImplementedError

    def provide_domains(
        self,
        *,
        ctx: NodeResolutionContext,
        resolved: PayoffMatrix | None = None,
    ) -> CharactersDomainObject:
        return CharactersDomainObject(characters=tuple(self.load_characters(ctx)))


@dataclass(frozen=True)
class CharacterInlineSource(CharacterSource):
    """インラインのキャラクター設定ソース。"""

    characters: tuple[CharacterNode, ...]

    def load_characters(self, ctx: NodeResolutionContext) -> list[Character]:
        return [
            Character(c.power, MatchupVector(c.vector[0], c.vector[1]), c.label)
            for c in self.characters
        ]


@dataclass(frozen=True)
class CharacterListFromFileNode(CharacterSource):
    """ファイルからキャラクターリストを読み込む設定ノード。

    NodeResolutionContext の load_characters_from_file を呼び出して解決する。
    """

    path: str

    def load_characters(self, ctx: NodeResolutionContext) -> list[Character]:
        return ctx.load_characters_from_file(self.path)


# ---------------------------------------------------------------------------
# Team nodes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TeamNode:
    """インラインのチーム設定ノード。"""

    label: str
    member_ids: tuple[str | int, ...]


class TeamSource(ApplicationNode[TeamsDomainObject]):
    """チーム入力ソースの抽象基底。"""

    @classmethod
    def from_node_spec(cls, spec: NodeSpec) -> TeamSource:
        """NodeSpec の refs または params.teams からチームソースを生成する。"""
        if "teams" in spec.refs:
            return TeamListFromFileNode(path=spec.refs["teams"])
        teams_data = spec.params.get("teams", [])
        return TeamInlineSource(
            teams=tuple(
                TeamNode(
                    label=t["label"],
                    member_ids=tuple(t["member_ids"]),
                )
                for t in teams_data
            )
        )

    @abstractmethod
    def load_teams(self, ctx: NodeResolutionContext) -> list[Team]:
        """チームリストを返す。"""
        raise NotImplementedError

    def provide_domains(
        self,
        *,
        ctx: NodeResolutionContext,
        resolved: PayoffMatrix | None = None,
    ) -> TeamsDomainObject:
        return TeamsDomainObject(teams=tuple(self.load_teams(ctx)))


@dataclass(frozen=True)
class TeamInlineSource(TeamSource):
    """インラインのチーム設定ソース。"""

    teams: tuple[TeamNode, ...]

    def load_teams(self, ctx: NodeResolutionContext) -> list[Team]:
        return [Team(label=t.label, member_ids=t.member_ids) for t in self.teams]


@dataclass(frozen=True)
class TeamListFromFileNode(TeamSource):
    """ファイルからチームリストを読み込む設定ノード。

    NodeResolutionContext の load_teams_from_file を呼び出して解決する。
    """

    path: str

    def load_teams(self, ctx: NodeResolutionContext) -> list[Team]:
        return ctx.load_teams_from_file(self.path)


# ---------------------------------------------------------------------------
# Output nodes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OutputEmission:
    """OutputNode が runner へ送る出力イベント。"""

    output_node: "OutputNode"
    node_name: str
    node_path: tuple[str, ...]
    runner: str | None
    domains: NodeDomainObject


class OutputNode(ABC):
    """出力ノードの抽象基底。

    出力種別を追加するには:
      1. OutputNode を継承し class 宣言に `output_method="..."` を指定する
      2. `_from_output_spec(cls, spec)` classmethod を実装する
    """

    _output_registry: ClassVar[dict[str, type[OutputNode]]] = {}
    _output_method: ClassVar[str] = ""

    def __init_subclass__(cls, output_method: str | None = None, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if output_method is not None:
            cls._output_method = output_method
            OutputNode._output_registry[output_method] = cls

    @classmethod
    def create_from_spec(cls, spec: OutputSpec) -> OutputNode:
        """OutputSpec から対応する OutputNode を生成して返す。"""
        node_cls = cls._output_registry.get(spec.method)
        if node_cls is None:
            raise ValueError(f"未知の output method です: {spec.method!r}")
        return node_cls._from_output_spec(spec)

    @classmethod
    def create_all_from_specs(cls, specs: tuple[OutputSpec, ...]) -> tuple[OutputNode, ...]:
        """OutputSpec のタプルを OutputNode のタプルに変換する。"""
        return tuple(cls.create_from_spec(s) for s in specs)

    @classmethod
    @abstractmethod
    def _from_output_spec(cls, spec: OutputSpec) -> OutputNode:
        """OutputSpec からこの出力種別のインスタンスを生成する。"""
        raise NotImplementedError

    @abstractmethod
    def emit(
        self,
        *,
        node_name: str,
        node_path: tuple[str, ...],
        domains: NodeDomainObject,
    ) -> OutputEmission:
        """Runner に渡す出力イベントを生成する。"""
        raise NotImplementedError

    @property
    def output_method(self) -> str:
        return self._output_method

    def resolve_runner(self) -> str | None:
        return getattr(self, "runner", None)

    @abstractmethod
    def execute(
        self,
        *,
        output_path_port: OutputPathPort,
        run_id: str,
        node_path: tuple[str, ...],
        domains: NodeDomainObject,
    ) -> Path:
        """Runner から呼び出され、最終成果物を生成する。"""
        raise NotImplementedError


def _extract_matrix(domains: NodeDomainObject) -> PayoffMatrix | None:
    if isinstance(
        domains,
        (
            MatrixDomainObject,
            MatrixCharactersDomainObject,
            MatrixTeamsDomainObject,
            MatrixCharactersTeamsDomainObject,
        ),
    ):
        return domains.matrix
    return None


def _extract_characters(domains: NodeDomainObject) -> tuple[Character, ...]:
    if isinstance(domains, (CharactersDomainObject, MatrixCharactersDomainObject)):
        return domains.characters
    if isinstance(domains, MatrixCharactersTeamsDomainObject):
        return domains.characters
    return ()


@dataclass(frozen=True)
class PayoffDirectedGraphOutputNode(OutputNode, output_method="payoff_directed_graph"):
    """有向グラフ出力設定ノード。"""

    runner: str | None = None
    filename: str = "payoff_directed_graph.svg"
    threshold: float = 0.0
    canvas_size: int = 840

    @classmethod
    def _from_output_spec(cls, spec: OutputSpec) -> PayoffDirectedGraphOutputNode:
        return cls(
            runner=spec.runner,
            filename=spec.params.get("filename", "payoff_directed_graph.svg"),
            threshold=spec.params.get("threshold", 0.0),
            canvas_size=spec.params.get("canvas_size", 840),
        )

    def emit(
        self,
        *,
        node_name: str,
        node_path: tuple[str, ...],
        domains: NodeDomainObject,
    ) -> OutputEmission:
        return OutputEmission(
            output_node=self,
            node_name=node_name,
            node_path=node_path,
            runner=self.runner,
            domains=domains,
        )

    def execute(
        self,
        *,
        output_path_port: OutputPathPort,
        run_id: str,
        node_path: tuple[str, ...],
        domains: NodeDomainObject,
    ) -> Path:
        matrix = _extract_matrix(domains)
        if matrix is None:
            raise ValueError("payoff_directed_graph は matrix を持つノードでのみ使用できます")
        path = output_path_port.resolve_output_path(
            run_id=run_id,
            node_path=node_path,
            output_method=self.output_method,
            filename=self.filename,
        )
        PayoffDirectedGraphPlotter(
            payoff_matrix=matrix.matrix,
            labels=matrix.labels,
            threshold=self.threshold,
        ).draw(path, canvas_size=self.canvas_size)
        return path


@dataclass(frozen=True)
class CharacterVectorGraphOutputNode(OutputNode, output_method="character_vector_graph"):
    """キャラクターベクトルグラフ出力設定ノード。"""

    runner: str | None = None
    filename: str = "character_vector_graph.svg"
    canvas_size: int = 840
    margin: int = 90

    @classmethod
    def _from_output_spec(cls, spec: OutputSpec) -> CharacterVectorGraphOutputNode:
        return cls(
            runner=spec.runner,
            filename=spec.params.get("filename", "character_vector_graph.svg"),
            canvas_size=spec.params.get("canvas_size", 840),
            margin=spec.params.get("margin", 90),
        )

    def emit(
        self,
        *,
        node_name: str,
        node_path: tuple[str, ...],
        domains: NodeDomainObject,
    ) -> OutputEmission:
        return OutputEmission(
            output_node=self,
            node_name=node_name,
            node_path=node_path,
            runner=self.runner,
            domains=domains,
        )

    def execute(
        self,
        *,
        output_path_port: OutputPathPort,
        run_id: str,
        node_path: tuple[str, ...],
        domains: NodeDomainObject,
    ) -> Path:
        path = output_path_port.resolve_output_path(
            run_id=run_id,
            node_path=node_path,
            output_method=self.output_method,
            filename=self.filename,
        )
        characters = _extract_characters(domains)
        if not characters:
            raise ValueError(
                "character_vector_graph は characters を持つノードでのみ使用できます"
            )
        CharacterVectorGraphPlotter(list(characters)).draw(
            output_path=path,
            canvas_size=self.canvas_size,
            margin=self.margin,
        )
        return path


@dataclass(frozen=True)
class EquilibriumOutputNode(OutputNode, output_method="equilibrium"):
    """均衡解ファイル出力設定ノード。"""

    runner: str | None = None
    filename: str = "equilibrium.toml"

    @classmethod
    def _from_output_spec(cls, spec: OutputSpec) -> EquilibriumOutputNode:
        return cls(
            runner=spec.runner,
            filename=spec.params.get("filename", "equilibrium.toml"),
        )

    def emit(
        self,
        *,
        node_name: str,
        node_path: tuple[str, ...],
        domains: NodeDomainObject,
    ) -> OutputEmission:
        return OutputEmission(
            output_node=self,
            node_name=node_name,
            node_path=node_path,
            runner=self.runner,
            domains=domains,
        )

    def execute(
        self,
        *,
        output_path_port: OutputPathPort,
        run_id: str,
        node_path: tuple[str, ...],
        domains: NodeDomainObject,
    ) -> Path:
        matrix = _extract_matrix(domains)
        if matrix is None:
            raise ValueError("equilibrium は matrix を持つノードでのみ使用できます")
        path = output_path_port.resolve_output_path(
            run_id=run_id,
            node_path=node_path,
            output_method=self.output_method,
            filename=self.filename,
        )
        mixed = SolverSelector().solve(matrix)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            tomli_w.dump(
                {
                    "strategies": [
                        {"id": sid, "probability": float(prob)}
                        for sid, prob in zip(mixed.strategy_ids, mixed.probabilities, strict=True)
                    ],
                },
                f,
            )
        return path


# ---------------------------------------------------------------------------
# MatrixNode — 行列構築ノードの抽象基底
# ---------------------------------------------------------------------------


class MatrixNode(ApplicationNode[NodeDomainObject]):
    """行列構築ノードの抽象基底。

    すべての具象ノードは name・outputs フィールドと build メソッドを実装する。

    ノード種別を追加するには:
      1. MatrixNode を継承し class 宣言に `node_method="..."` を指定する
      2. `_from_spec(cls, spec, build_child)` classmethod を実装する
      3. `build(self, ctx)` を実装する
      以上のみ。MatrixNodeFactory への変更は不要。
    """

    _node_registry: ClassVar[dict[str, type[MatrixNode]]] = {}

    name: str
    outputs: tuple[OutputNode, ...]

    def __init_subclass__(cls, node_method: str | None = None, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if node_method is not None:
            MatrixNode._node_registry[node_method] = cls

    @classmethod
    def create_from_spec(
        cls,
        spec: NodeSpec,
        build_child: Callable[[NodeSpec], MatrixNode],
    ) -> MatrixNode:
        """NodeSpec から対応する MatrixNode を生成して返す。"""
        node_cls = cls._node_registry.get(spec.method)
        if node_cls is None:
            raise ValueError(f"未知の method です: {spec.method!r}")
        return node_cls._from_spec(spec, build_child)

    @classmethod
    @abstractmethod
    def _from_spec(
        cls,
        spec: NodeSpec,
        build_child: Callable[[NodeSpec], MatrixNode],
    ) -> MatrixNode:
        """NodeSpec からこのノード種別のインスタンスを生成する。"""
        raise NotImplementedError

    @abstractmethod
    def build(self, ctx: NodeResolutionContext) -> PayoffMatrix:
        """コンテキストを使って PayoffMatrix を構築して返す。"""
        raise NotImplementedError

    def provide_domains(
        self,
        *,
        ctx: NodeResolutionContext,
        resolved: PayoffMatrix | None = None,
    ) -> MatrixDomainObject:
        """解決済み結果から出力連携用ドメインオブジェクトを返す。"""
        if resolved is None:
            raise ValueError("行列構築ノードの domain 提供には resolved matrix が必要です")
        return MatrixDomainObject(matrix=resolved)


# ---------------------------------------------------------------------------
# Matrix nodes — 各生成方式ごとの具象ノード
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GeneralFromRawNode(MatrixNode, node_method="general_from_raw"):
    """生行列データから一般利得行列を構築するノード。"""

    matrix: Any  # list[list[float]] または np.ndarray
    labels: list[str] | None = None
    name: str = "root"
    outputs: tuple[OutputNode, ...] = field(default_factory=tuple)

    @classmethod
    def _from_spec(
        cls, spec: NodeSpec, build_child: Callable[[NodeSpec], MatrixNode]
    ) -> GeneralFromRawNode:
        return cls(
            matrix=spec.params["matrix"],
            labels=spec.params.get("labels"),
            name=spec.name,
            outputs=OutputNode.create_all_from_specs(spec.outputs),
        )

    def build(self, ctx: NodeResolutionContext) -> PayoffMatrix:
        matrix = np.asarray(self.matrix, dtype=float)
        return PayoffMatrixBuilder.from_general_matrix(matrix=matrix, labels=self.labels)


@dataclass(frozen=True)
class MonocycleFromCharactersNode(MatrixNode, node_method="monocycle_from_characters"):
    """キャラクターリストから単相性モデル利得行列を構築するノード。

    characters には CharacterInlineSource または CharacterListFromFileNode を指定する。
    """

    characters: CharacterSource
    labels: list[str] | None = None
    name: str = "root"
    outputs: tuple[OutputNode, ...] = field(default_factory=tuple)

    @classmethod
    def _from_spec(
        cls, spec: NodeSpec, build_child: Callable[[NodeSpec], MatrixNode]
    ) -> MonocycleFromCharactersNode:
        return cls(
            characters=CharacterSource.from_node_spec(spec),
            labels=spec.params.get("labels"),
            name=spec.name,
            outputs=OutputNode.create_all_from_specs(spec.outputs),
        )

    def build(self, ctx: NodeResolutionContext) -> PayoffMatrix:
        character_domains = ctx.resolve_node_domains(self.characters)
        return PayoffMatrixBuilder.from_characters(
            characters=list(character_domains.characters), labels=self.labels
        )

    def provide_domains(
        self,
        *,
        ctx: NodeResolutionContext,
        resolved: PayoffMatrix | None = None,
    ) -> MatrixCharactersDomainObject:
        base_domains = super().provide_domains(ctx=ctx, resolved=resolved)
        character_domains = self.characters.provide_domains(ctx=ctx)
        return MatrixCharactersDomainObject(
            matrix=base_domains.matrix,
            characters=character_domains.characters,
        )


@dataclass(frozen=True)
class GeneralFromTeamsPayoffNode(MatrixNode, node_method="general_from_teams_payoff"):
    """計算済みチーム利得行列から一般利得行列を構築するノード。

    teams には TeamInlineSource または TeamListFromFileNode を指定する。
    """

    team_payoff: Any  # list[list[float]] または np.ndarray
    teams: TeamSource
    name: str = "root"
    outputs: tuple[OutputNode, ...] = field(default_factory=tuple)

    @classmethod
    def _from_spec(
        cls, spec: NodeSpec, build_child: Callable[[NodeSpec], MatrixNode]
    ) -> GeneralFromTeamsPayoffNode:
        return cls(
            team_payoff=spec.params["team_payoff"],
            teams=TeamSource.from_node_spec(spec),
            name=spec.name,
            outputs=OutputNode.create_all_from_specs(spec.outputs),
        )

    def build(self, ctx: NodeResolutionContext) -> PayoffMatrix:
        team_payoff = np.asarray(self.team_payoff, dtype=float)
        team_domains = ctx.resolve_node_domains(self.teams)
        return PayoffMatrixBuilder.from_teams(team_payoff=team_payoff, teams=list(team_domains.teams))

    def provide_domains(
        self,
        *,
        ctx: NodeResolutionContext,
        resolved: PayoffMatrix | None = None,
    ) -> MatrixTeamsDomainObject:
        base_domains = super().provide_domains(ctx=ctx, resolved=resolved)
        team_domains = self.teams.provide_domains(ctx=ctx)
        return MatrixTeamsDomainObject(
            matrix=base_domains.matrix,
            teams=team_domains.teams,
        )


@dataclass(frozen=True)
class GeneralFromTeamMatchupsNode(MatrixNode, node_method="general_from_team_matchups"):
    """キャラクター行列とチーム定義からチーム利得行列を構築するノード。

    character_matrix には任意の MatrixNode を再帰的に指定できる。
    teams には TeamInlineSource または TeamListFromFileNode を指定する。
    """

    teams: TeamSource
    character_matrix: MatrixNode
    use_monocycle_formula: bool = True
    name: str = "root"
    outputs: tuple[OutputNode, ...] = field(default_factory=tuple)

    @classmethod
    def _from_spec(
        cls, spec: NodeSpec, build_child: Callable[[NodeSpec], MatrixNode]
    ) -> GeneralFromTeamMatchupsNode:
        character_matrix_spec = spec.children.get("character_matrix")
        if character_matrix_spec is None:
            raise ValueError(
                "general_from_team_matchups には children.character_matrix が必要です"
            )
        return cls(
            teams=TeamSource.from_node_spec(spec),
            character_matrix=build_child(character_matrix_spec),
            use_monocycle_formula=spec.params.get("use_monocycle_formula", True),
            name=spec.name,
            outputs=OutputNode.create_all_from_specs(spec.outputs),
        )

    def build(self, ctx: NodeResolutionContext) -> PayoffMatrix:
        character_matrix = ctx.resolve_node(self.character_matrix)
        team_domains = ctx.resolve_node_domains(self.teams)
        return PayoffMatrixBuilder.from_team_matchups(
            teams=list(team_domains.teams),
            character_matrix=character_matrix,
            use_monocycle_formula=self.use_monocycle_formula,
        )

    def provide_domains(
        self,
        *,
        ctx: NodeResolutionContext,
        resolved: PayoffMatrix | None = None,
    ) -> MatrixCharactersTeamsDomainObject:
        base_domains = super().provide_domains(ctx=ctx, resolved=resolved)
        team_domains = self.teams.provide_domains(ctx=ctx)
        child_domains = ctx.resolve_node_domains(self.character_matrix)
        child_characters = _extract_characters(child_domains)
        return MatrixCharactersTeamsDomainObject(
            matrix=base_domains.matrix,
            characters=child_characters,
            teams=team_domains.teams,
        )


@dataclass(frozen=True)
class RandomSkewSymmetricNode(MatrixNode, node_method="random_skew_symmetric"):
    """ランダム交代行列を生成するノード。"""

    size: int
    low: float = -1.0
    high: float = 1.0
    seed: int | None = None
    max_attempts: int = 10_000
    labels: list[str] | None = None
    name: str = "root"
    outputs: tuple[OutputNode, ...] = field(default_factory=tuple)

    @classmethod
    def _from_spec(
        cls, spec: NodeSpec, build_child: Callable[[NodeSpec], MatrixNode]
    ) -> RandomSkewSymmetricNode:
        return cls(
            size=spec.params["size"],
            low=spec.params.get("low", -1.0),
            high=spec.params.get("high", 1.0),
            seed=spec.params.get("seed"),
            max_attempts=spec.params.get("max_attempts", 10_000),
            labels=spec.params.get("labels"),
            name=spec.name,
            outputs=OutputNode.create_all_from_specs(spec.outputs),
        )

    def build(self, ctx: NodeResolutionContext) -> PayoffMatrix:
        rng = np.random.default_rng(self.seed) if self.seed is not None else None
        return PayoffMatrixBuilder.from_random_matrix(
            size=self.size,
            low=self.low,
            high=self.high,
            rng=rng,
            max_attempts=self.max_attempts,
            labels=self.labels,
        )


@dataclass(frozen=True)
class ApproxMonocycleToGeneralNode(MatrixNode, node_method="approx_monocycle_to_general"):
    """単相性行列を一般行列へ変換する近似ノード。

    source には任意の MatrixNode を再帰的に指定できる。
    """

    source: MatrixNode
    name: str = "root"
    outputs: tuple[OutputNode, ...] = field(default_factory=tuple)

    @classmethod
    def _from_spec(
        cls, spec: NodeSpec, build_child: Callable[[NodeSpec], MatrixNode]
    ) -> ApproxMonocycleToGeneralNode:
        source_spec = spec.children.get("source")
        if source_spec is None:
            raise ValueError("approx_monocycle_to_general には children.source が必要です")
        return cls(
            source=build_child(source_spec),
            name=spec.name,
            outputs=OutputNode.create_all_from_specs(spec.outputs),
        )

    def build(self, ctx: NodeResolutionContext) -> PayoffMatrix:
        source = ctx.resolve_node(self.source)
        return MonocycleToGeneralApproximation().approximate(source).matrix


@dataclass(frozen=True)
class ApproxDominantEigenpairNode(MatrixNode, node_method="approx_dominant_eigenpair"):
    """支配固有値ペアによる近似変換ノード。

    source には任意の MatrixNode を再帰的に指定できる。
    """

    source: MatrixNode
    atol: float = 1e-8
    name: str = "root"
    outputs: tuple[OutputNode, ...] = field(default_factory=tuple)

    @classmethod
    def _from_spec(
        cls, spec: NodeSpec, build_child: Callable[[NodeSpec], MatrixNode]
    ) -> ApproxDominantEigenpairNode:
        source_spec = spec.children.get("source")
        if source_spec is None:
            raise ValueError("approx_dominant_eigenpair には children.source が必要です")
        return cls(
            source=build_child(source_spec),
            atol=spec.params.get("atol", 1e-8),
            name=spec.name,
            outputs=OutputNode.create_all_from_specs(spec.outputs),
        )

    def build(self, ctx: NodeResolutionContext) -> PayoffMatrix:
        source = ctx.resolve_node(self.source)
        return DominantEigenpairMonocycleApproximation(atol=self.atol).approximate(source).matrix


@dataclass(frozen=True)
class ApproxEquilibriumPreservingNode(MatrixNode, node_method="approx_equilibrium_preserving"):
    """均衡保存残差近似変換ノード。

    source には任意の MatrixNode を再帰的に指定できる。
    """

    source: MatrixNode
    atol: float = 1e-8
    name: str = "root"
    outputs: tuple[OutputNode, ...] = field(default_factory=tuple)

    @classmethod
    def _from_spec(
        cls, spec: NodeSpec, build_child: Callable[[NodeSpec], MatrixNode]
    ) -> ApproxEquilibriumPreservingNode:
        source_spec = spec.children.get("source")
        if source_spec is None:
            raise ValueError("approx_equilibrium_preserving には children.source が必要です")
        return cls(
            source=build_child(source_spec),
            atol=spec.params.get("atol", 1e-8),
            name=spec.name,
            outputs=OutputNode.create_all_from_specs(spec.outputs),
        )

    def build(self, ctx: NodeResolutionContext) -> PayoffMatrix:
        source = ctx.resolve_node(self.source)
        return EquilibriumPreservingResidualMonocycleApproximation(atol=self.atol).approximate(source).matrix
