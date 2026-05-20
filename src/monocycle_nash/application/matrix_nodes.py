"""Application-layer typed node definitions for payoff matrix construction.

各ノード型が「この生成方式はこういう値や設定を受け取る」を明示する。
依存する他ドメインモデルは型付きフィールドとして直接保持し、
各ノードは resolve_value / emit / execute / load_characters / load_teams で
それぞれの解決ロジックを担う。

新規ノード種別を追加する場合は:
  1. 具象 MatrixNode サブクラスを作り、class 宣言に `node_method="..."` を付ける
  2. `_from_spec(cls, spec, build_child)` classmethod を実装する
  以上のみ。MatrixNodeFactory や登録処理への変更は不要。

新規出力種別を追加する場合は:
  1. 具象 OutputNode サブクラスを作り、class 宣言に `output_method="..."` を付ける
  2. `_from_output_spec(cls, spec)` classmethod を実装する
  3. `emit(...)` と `execute(...)` を実装する
  4. 必要に応じて `execute_emissions(...)` を実装し、複数 emit の集約実行を行う
"""

from __future__ import annotations

import math
import numpy as np
import tomli_w
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, ClassVar, Generic, TypeVar, cast

from monocycle_nash.application.node_spec import NodeSpec, OutputSpec
from monocycle_nash.application.ports import (
    CharacterListFilePort,
    OutputPathPort,
    TeamListFilePort,
)
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


DomainT = TypeVar("DomainT")
PayloadT = TypeVar("PayloadT")
# TOML 由来のネスト配列か、既に数値化済みの ndarray を受け付ける。
RawMatrix = list[list[float]] | np.ndarray
FULL_ROTATION_RAD = 2 * math.pi


# ---------------------------------------------------------------------------
# NodeResolutionContext — ノードが使用するコンテキストインターフェース
# ---------------------------------------------------------------------------


class NodeResolutionContext(ABC):
    """ApplicationNode 解決時に再帰参照で使うコンテキスト。"""

    @abstractmethod
    def resolve_node(self, node: "ApplicationNode[DomainT]") -> "ApplicationNode[DomainT]":
        """別ノードを再帰的に解決してノード自身を返す。"""
        raise NotImplementedError

    @abstractmethod
    def get_node_value(self, node: "ApplicationNode[DomainT]") -> DomainT:
        """解決済みノードの value を返す。"""
        raise NotImplementedError

# ---------------------------------------------------------------------------
# ApplicationNode — 全ノード共通の抽象基底
# ---------------------------------------------------------------------------


class ApplicationNode(ABC, Generic[DomainT]):
    """解決済みオブジェクトを提供する全ノード共通抽象。"""

    # MatrixConfigTreeResolver による解決後に設定されるノード値。
    # 解決前にアクセスした場合の挙動は未定義。
    value: DomainT
    _is_resolved: bool = False

    def set_value(self, value: DomainT) -> None:
        """解決済み value をノードへ設定する。"""
        self.value = value
        self._is_resolved = True

    def is_resolved(self) -> bool:
        """このノードが解決済みかどうかを返す。"""
        return self._is_resolved

    def resolve_value(
        self,
        *,
        ctx: NodeResolutionContext,
    ) -> DomainT:
        """resolver がこのノードを評価するときの値計算処理。"""
        return self.provide_object(ctx=ctx)

    def resolve_name(self) -> str | None:
        """resolver 上のノード名。name 属性がある場合のみ返す。"""
        name = getattr(self, "name", None)
        if name is None or isinstance(name, str):
            return name
        raise TypeError("ApplicationNode.name は str である必要があります")

    def resolve_outputs(self) -> tuple["OutputNode", ...]:
        """resolver 上の出力ノード列。outputs 属性がある場合のみ返す。"""
        outputs = getattr(self, "outputs", ())
        resolved_outputs: list["OutputNode"] = []
        for output in outputs:
            if not isinstance(output, OutputNode):
                raise TypeError("ApplicationNode.outputs は OutputNode の列である必要があります")
            resolved_outputs.append(output)
        return tuple(resolved_outputs)

    @abstractmethod
    def provide_object(
        self,
        *,
        ctx: NodeResolutionContext,
    ) -> DomainT:
        """ノードが提供するオブジェクトを返す。"""
        raise NotImplementedError


NodeT = TypeVar("NodeT", bound=ApplicationNode)


# ---------------------------------------------------------------------------
# Character nodes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CharacterNode:
    """インラインのキャラクター設定ノード。"""

    power: float
    vector: tuple[float, float]
    label: str = ""


class CharacterSource(ApplicationNode[tuple[Character, ...]]):
    """キャラクター入力ソースの抽象基底。"""

    _registry: ClassVar[dict[str, type["CharacterSource"]]] = {}
    node_method: ClassVar[str]

    def __init_subclass__(cls, **kwargs: object) -> None:
        node_method = kwargs.pop("node_method", None)
        super().__init_subclass__(**kwargs)
        if node_method is None:
            return
        if not isinstance(node_method, str):
            raise TypeError("CharacterSource の node_method は str である必要があります")
        cls.node_method = node_method
        CharacterSource._registry[node_method] = cls

    @classmethod
    def create_from_spec(cls, spec: NodeSpec) -> "CharacterSource":
        node_cls = cls._registry.get(spec.method)
        if node_cls is None:
            raise ValueError(f"未知の character source method です: {spec.method}")
        return node_cls._from_spec(spec)

    @classmethod
    def from_node_spec(cls, spec: NodeSpec) -> CharacterSource:
        """NodeSpec からキャラクターソースを生成する。"""
        characters_spec = spec.children.get("characters")
        if characters_spec is not None:
            return cls.create_from_spec(characters_spec)

        # 後方互換: 既存の params/refs 直書きにも対応する。
        if "characters" in spec.refs:
            return CharacterListFromFileNode(
                path=spec.refs["characters"],
                name="characters",
            )
        chars_data = spec.params.get("characters", [])
        return CharacterInlineSource(
            characters=tuple(
                CharacterNode(
                    power=c["power"],
                    vector=(float(c["vector"][0]), float(c["vector"][1])),
                    label=c.get("label", ""),
                )
                for c in chars_data
            ),
            name="characters",
        )

    @classmethod
    @abstractmethod
    def _from_spec(cls, spec: NodeSpec) -> "CharacterSource":
        raise NotImplementedError

    @abstractmethod
    def load_characters(self, ctx: NodeResolutionContext) -> tuple[Character, ...]:
        """キャラクターリストを返す。"""
        raise NotImplementedError

    def provide_object(
        self,
        *,
        ctx: NodeResolutionContext,
    ) -> tuple[Character, ...]:
        return self.load_characters(ctx)


@dataclass
class CharacterInlineSource(CharacterSource, node_method="character_inline"):
    """インラインのキャラクター設定ソース。"""

    characters: tuple[CharacterNode, ...]
    name: str = "characters"
    outputs: tuple["OutputNode", ...] = field(default_factory=tuple)

    @classmethod
    def _from_spec(cls, spec: NodeSpec) -> "CharacterInlineSource":
        chars_data = spec.params.get("characters", [])
        return cls(
            characters=tuple(
                CharacterNode(
                    power=c["power"],
                    vector=(float(c["vector"][0]), float(c["vector"][1])),
                    label=c.get("label", ""),
                )
                for c in chars_data
            ),
            name=spec.name,
            outputs=OutputNode.create_all_from_specs(spec.outputs),
        )

    def load_characters(self, ctx: NodeResolutionContext) -> tuple[Character, ...]:
        return tuple(
            Character(c.power, MatchupVector(c.vector[0], c.vector[1]), c.label)
            for c in self.characters
        )


@dataclass
class CharacterRotatingPairSource(CharacterSource, node_method="character_rotating_pair"):
    """回転する2点ペアを含むキャラクター列を生成する設定ソース。"""

    x: float
    y: float
    r3: float
    r4: float
    d: float
    theta_step_rad: float
    power: float = 0.0
    fixed_label_1: str = "c1"
    fixed_label_2: str = "c2"
    rotating_label_3_prefix: str = "g3_"
    rotating_label_4_prefix: str = "g4_"
    index_width: int = 3
    theta_offset_rad: float = 0.0
    name: str = "characters"
    outputs: tuple["OutputNode", ...] = field(default_factory=tuple)

    @classmethod
    def _from_spec(cls, spec: NodeSpec) -> "CharacterRotatingPairSource":
        theta_step_rad = spec.params.get("theta_step_rad")
        theta_step_deg = spec.params.get("theta_step_deg")
        if theta_step_rad is not None and theta_step_deg is not None:
            raise ValueError(
                "character_rotating_pair では theta_step_rad と theta_step_deg を同時に指定できません"
            )
        if theta_step_rad is None:
            if theta_step_deg is None:
                raise ValueError(
                    "character_rotating_pair には theta_step_rad または theta_step_deg が必要です"
                )
            theta_step_rad = math.radians(float(theta_step_deg))
        theta_step_rad = float(theta_step_rad)
        if theta_step_rad <= 0:
            raise ValueError("character_rotating_pair の theta_step は正の値が必要です")
        if theta_step_rad > FULL_ROTATION_RAD:
            raise ValueError("character_rotating_pair の theta_step は 2π 以下で指定してください")

        return cls(
            x=float(spec.params["x"]),
            y=float(spec.params["y"]),
            r3=float(spec.params["r3"]),
            r4=float(spec.params["r4"]),
            d=float(spec.params["d"]),
            theta_step_rad=theta_step_rad,
            power=float(spec.params.get("power", 0.0)),
            fixed_label_1=str(spec.params.get("fixed_label_1", "c1")),
            fixed_label_2=str(spec.params.get("fixed_label_2", "c2")),
            rotating_label_3_prefix=str(spec.params.get("rotating_label_3_prefix", "g3_")),
            rotating_label_4_prefix=str(spec.params.get("rotating_label_4_prefix", "g4_")),
            index_width=int(spec.params.get("index_width", 3)),
            theta_offset_rad=float(spec.params.get("theta_offset_rad", 0.0)),
            name=spec.name,
            outputs=OutputNode.create_all_from_specs(spec.outputs),
        )

    def load_characters(self, ctx: NodeResolutionContext) -> tuple[Character, ...]:
        count = int(round(FULL_ROTATION_RAD / self.theta_step_rad))
        if count < 1:
            raise ValueError("character_rotating_pair で生成される点数が0です")

        characters: list[Character] = [
            Character(
                self.power,
                MatchupVector(float(self.x), 0.0),
                self.fixed_label_1,
            ),
            Character(
                self.power,
                MatchupVector(0.0, float(self.y)),
                self.fixed_label_2,
            ),
        ]
        for i in range(count):
            suffix = f"{i:0{self.index_width}d}"
            theta = self.theta_offset_rad + i * self.theta_step_rad
            sin_theta = math.sin(theta)
            cos_theta = math.cos(theta)
            sin_d_theta = math.sin(self.d + theta)
            cos_d_theta = math.cos(self.d + theta)
            # Match the problem definition convention: v=(r*sinθ, r*cosθ).
            v3 = MatchupVector(self.r3 * sin_theta, self.r3 * cos_theta)
            v4 = MatchupVector(self.r4 * sin_d_theta, self.r4 * cos_d_theta)
            characters.append(Character(self.power, v3, f"{self.rotating_label_3_prefix}{suffix}"))
            characters.append(Character(self.power, v4, f"{self.rotating_label_4_prefix}{suffix}"))
        return tuple(characters)


@dataclass
class CharacterListFromFileNode(CharacterSource, node_method="character_from_file"):
    """ファイルからキャラクターリストを読み込む設定ノード。

    CharacterListFilePort を用いて解決する。
    """

    path: str
    character_list_file_port: CharacterListFilePort | None = None
    name: str = "characters"
    outputs: tuple["OutputNode", ...] = field(default_factory=tuple)

    @classmethod
    def _from_spec(cls, spec: NodeSpec) -> "CharacterListFromFileNode":
        refs_path = spec.refs.get("characters")
        if refs_path is None:
            raise ValueError("character_from_file には refs.characters が必要です")
        return cls(
            path=refs_path,
            name=spec.name,
            outputs=OutputNode.create_all_from_specs(spec.outputs),
        )

    def load_characters(self, ctx: NodeResolutionContext) -> tuple[Character, ...]:
        if self.character_list_file_port is None:
            raise ValueError(
                "CharacterListFromFileNode を解決するには CharacterListFilePort が必要です"
            )
        return self.character_list_file_port.load_characters(self.path)


# ---------------------------------------------------------------------------
# Team nodes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TeamNode:
    """インラインのチーム設定ノード。"""

    label: str
    member_ids: tuple[str | int, ...]


class TeamSource(ApplicationNode[tuple[Team, ...]]):
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
    def load_teams(self, ctx: NodeResolutionContext) -> tuple[Team, ...]:
        """チームリストを返す。"""
        raise NotImplementedError

    def get_teams(self, *, ctx: NodeResolutionContext) -> tuple[Team, ...]:
        """チームタプルを返す（provide_object への便利メソッド）。"""
        return self.provide_object(ctx=ctx)

    def provide_object(
        self,
        *,
        ctx: NodeResolutionContext,
    ) -> tuple[Team, ...]:
        return self.load_teams(ctx)


@dataclass
class TeamInlineSource(TeamSource):
    """インラインのチーム設定ソース。"""

    teams: tuple[TeamNode, ...]

    def load_teams(self, ctx: NodeResolutionContext) -> tuple[Team, ...]:
        return tuple(Team(label=t.label, member_ids=t.member_ids) for t in self.teams)


@dataclass
class TeamListFromFileNode(TeamSource):
    """ファイルからチームリストを読み込む設定ノード。

    TeamListFilePort を用いて解決する。
    """

    path: str
    team_list_file_port: TeamListFilePort | None = None

    def load_teams(self, ctx: NodeResolutionContext) -> tuple[Team, ...]:
        if self.team_list_file_port is None:
            raise ValueError("TeamListFromFileNode を解決するには TeamListFilePort が必要です")
        return self.team_list_file_port.load_teams(self.path)


# ---------------------------------------------------------------------------
# Output nodes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OutputEmission(Generic[DomainT, PayloadT]):
    """OutputNode が runner へ送る出力イベント。

    payload は OutputNode 実装が任意で持ち回る補助メタデータ。
    型は各 OutputNode 実装側で `emit()` の返却時に具体化する。
    """

    output_node: "OutputNode"
    node_name: str
    node_path: tuple[str, ...]
    runner: str | None
    node: "ApplicationNode[DomainT]"
    payload: PayloadT | None = None

    def resolve_value(self, *, ctx: "NodeResolutionContext") -> DomainT:
        """Resolve and return the domain object provided by this emission's node."""
        return ctx.get_node_value(self.node)


class OutputNode(ABC, Generic[NodeT]):
    """出力ノードの抽象基底。

    出力種別を追加するには:
      1. OutputNode を継承し class 宣言に `output_method="..."` を指定する
      2. `_from_output_spec(cls, spec)` classmethod を実装する
    """

    _output_registry: ClassVar[dict[str, type[OutputNode]]] = {}
    _output_method: ClassVar[str] = ""

    def __init_subclass__(cls, output_method: str | None = None) -> None:
        super().__init_subclass__()
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
        node: NodeT,
    ) -> OutputEmission:
        """Runner に渡す出力イベントを生成する。"""
        raise NotImplementedError

    @property
    def output_method(self) -> str:
        return self._output_method

    def resolve_runner(self) -> str | None:
        return getattr(self, "runner", None)

    def execute_emissions(
        self,
        *,
        output_path_port: OutputPathPort,
        run_id: str,
        emissions: tuple[OutputEmission, ...],
        ctx: NodeResolutionContext,
    ) -> tuple[Path, ...]:
        """Execute runner emissions and return output artifact paths."""
        return tuple(
            emission.output_node.execute(
                output_path_port=output_path_port,
                run_id=run_id,
                node_path=emission.node_path,
                node=cast(NodeT, emission.node),
                ctx=ctx,
            )
            for emission in emissions
        )

    @abstractmethod
    def execute(
        self,
        *,
        output_path_port: OutputPathPort,
        run_id: str,
        node_path: tuple[str, ...],
        node: NodeT,
        ctx: NodeResolutionContext,
    ) -> Path:
        """Runner から呼び出され、最終成果物を生成する。"""
        raise NotImplementedError


@dataclass(frozen=True)
class PayoffDirectedGraphOutputNode(OutputNode["MatrixNode"], output_method="payoff_directed_graph"):
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
        node: "MatrixNode",
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
        node: "MatrixNode",
        ctx: NodeResolutionContext,
    ) -> Path:
        path = output_path_port.resolve_output_path(
            run_id=run_id,
            node_path=node_path,
            output_method=self.output_method,
            filename=self.filename,
        )
        matrix = node.provide_object(ctx=ctx)
        PayoffDirectedGraphPlotter(
            payoff_matrix=matrix.matrix,
            labels=matrix.labels,
            threshold=self.threshold,
        ).draw(path, canvas_size=self.canvas_size)
        return path


@dataclass(frozen=True)
class CharacterVectorGraphOutputNode(OutputNode["CharacterSource"], output_method="character_vector_graph"):
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
        node: "CharacterSource",
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
        node: "CharacterSource",
        ctx: NodeResolutionContext,
    ) -> Path:
        path = output_path_port.resolve_output_path(
            run_id=run_id,
            node_path=node_path,
            output_method=self.output_method,
            filename=self.filename,
        )
        characters = node.provide_object(ctx=ctx)
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
class EquilibriumOutputNode(OutputNode["MatrixNode"], output_method="equilibrium"):
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
        node: "MatrixNode",
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
        node: "MatrixNode",
        ctx: NodeResolutionContext,
    ) -> Path:
        path = output_path_port.resolve_output_path(
            run_id=run_id,
            node_path=node_path,
            output_method=self.output_method,
            filename=self.filename,
        )
        matrix = node.provide_object(ctx=ctx)
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


class MatrixNode(ApplicationNode[PayoffMatrix]):
    """行列構築ノードの抽象基底。

    すべての具象ノードは name・outputs フィールドと resolve_value メソッドを実装する。

    ノード種別を追加するには:
      1. MatrixNode を継承し class 宣言に `node_method="..."` を指定する
      2. `_from_spec(cls, spec, build_child)` classmethod を実装する
      3. `resolve_value(self, ctx)` を実装する
      以上のみ。MatrixNodeFactory への変更は不要。
    """

    _node_registry: ClassVar[dict[str, type[MatrixNode]]] = {}

    name: str
    outputs: tuple[OutputNode, ...]

    def __init_subclass__(cls, node_method: str | None = None) -> None:
        super().__init_subclass__()
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
    def resolve_value(
        self,
        *,
        ctx: NodeResolutionContext,
    ) -> PayoffMatrix:
        """コンテキストを使って PayoffMatrix を構築して返す。"""
        raise NotImplementedError

    def provide_object(
        self,
        *,
        ctx: NodeResolutionContext,
    ) -> PayoffMatrix:
        """解決済み結果の行列を返す。"""
        return cast(PayoffMatrix, ctx.get_node_value(self))


# ---------------------------------------------------------------------------
# Matrix nodes — 各生成方式ごとの具象ノード
# ---------------------------------------------------------------------------


@dataclass
class GeneralFromRawNode(MatrixNode, node_method="general_from_raw"):
    """生行列データから一般利得行列を構築するノード。"""

    matrix: RawMatrix
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

    def resolve_value(
        self,
        *,
        ctx: NodeResolutionContext,
    ) -> PayoffMatrix:
        matrix = np.asarray(self.matrix, dtype=float)
        return PayoffMatrixBuilder.from_general_matrix(matrix=matrix, labels=self.labels)


@dataclass
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

    def resolve_value(
        self,
        *,
        ctx: NodeResolutionContext,
    ) -> PayoffMatrix:
        characters = cast(tuple[Character, ...], ctx.get_node_value(self.characters))
        return PayoffMatrixBuilder.from_characters(characters=characters, labels=self.labels)


@dataclass
class GeneralFromTeamsPayoffNode(MatrixNode, node_method="general_from_teams_payoff"):
    """計算済みチーム利得行列から一般利得行列を構築するノード。

    teams には TeamInlineSource または TeamListFromFileNode を指定する。
    """

    team_payoff: RawMatrix
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

    def resolve_value(
        self,
        *,
        ctx: NodeResolutionContext,
    ) -> PayoffMatrix:
        team_payoff = np.asarray(self.team_payoff, dtype=float)
        teams = cast(tuple[Team, ...], ctx.get_node_value(self.teams))
        return PayoffMatrixBuilder.from_teams(team_payoff=team_payoff, teams=teams)


@dataclass
class GeneralFromTeamMatchupsNode(MatrixNode, node_method="general_from_team_matchups"):
    """キャラクター行列とチーム定義からチーム利得行列を構築するノード。

    characters には CharacterInlineSource または CharacterListFromFileNode を指定する。
    teams には TeamInlineSource または TeamListFromFileNode を指定する。
    """

    teams: TeamSource
    characters: CharacterSource
    use_monocycle_formula: bool = True
    name: str = "root"
    outputs: tuple[OutputNode, ...] = field(default_factory=tuple)

    @classmethod
    def _from_spec(
        cls, spec: NodeSpec, build_child: Callable[[NodeSpec], MatrixNode]
    ) -> GeneralFromTeamMatchupsNode:
        characters_spec = spec.children.get("characters")
        if characters_spec is None:
            raise ValueError("general_from_team_matchups には children.characters が必要です")
        return cls(
            teams=TeamSource.from_node_spec(spec),
            characters=CharacterSource.create_from_spec(characters_spec),
            use_monocycle_formula=spec.params.get("use_monocycle_formula", True),
            name=spec.name,
            outputs=OutputNode.create_all_from_specs(spec.outputs),
        )

    def resolve_value(
        self,
        *,
        ctx: NodeResolutionContext,
    ) -> PayoffMatrix:
        character_matrix = self.resolve_character_matrix(ctx=ctx)
        teams = cast(tuple[Team, ...], ctx.get_node_value(self.teams))
        return PayoffMatrixBuilder.from_team_matchups(
            teams=teams,
            character_matrix=character_matrix,
            use_monocycle_formula=self.use_monocycle_formula,
        )

    def resolve_character_matrix(
        self,
        *,
        ctx: NodeResolutionContext,
    ) -> PayoffMatrix:
        characters = cast(tuple[Character, ...], ctx.get_node_value(self.characters))
        labels = [c.label for c in characters]
        resolved_labels = labels if all(label != "" for label in labels) else None
        return PayoffMatrixBuilder.from_characters(
            characters=characters,
            labels=resolved_labels,
        )


@dataclass
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

    def resolve_value(
        self,
        *,
        ctx: NodeResolutionContext,
    ) -> PayoffMatrix:
        rng = np.random.default_rng(self.seed) if self.seed is not None else None
        return PayoffMatrixBuilder.from_random_matrix(
            size=self.size,
            low=self.low,
            high=self.high,
            rng=rng,
            max_attempts=self.max_attempts,
            labels=self.labels,
        )


@dataclass
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

    def resolve_value(
        self,
        *,
        ctx: NodeResolutionContext,
    ) -> PayoffMatrix:
        source_node = ctx.resolve_node(self.source)
        source = cast(PayoffMatrix, source_node.value)
        return MonocycleToGeneralApproximation().approximate(source).matrix


@dataclass
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

    def resolve_value(
        self,
        *,
        ctx: NodeResolutionContext,
    ) -> PayoffMatrix:
        source_node = ctx.resolve_node(self.source)
        source = cast(PayoffMatrix, source_node.value)
        return DominantEigenpairMonocycleApproximation(atol=self.atol).approximate(source).matrix


@dataclass
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

    def resolve_value(
        self,
        *,
        ctx: NodeResolutionContext,
    ) -> PayoffMatrix:
        source_node = ctx.resolve_node(self.source)
        source = cast(PayoffMatrix, source_node.value)
        return EquilibriumPreservingResidualMonocycleApproximation(atol=self.atol).approximate(source).matrix
