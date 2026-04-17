"""Application-layer typed node definitions for payoff matrix construction.

各ノード型が「この生成方式はこういう値や設定を受け取る」を明示する。
依存する他ドメインモデルは型付きフィールドとして直接保持し、
各ノードは build / run / load_characters / load_teams でそれぞれの解決ロジックを担う。
"""

from __future__ import annotations

import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from monocycle_nash.application.ports import OutputPathPort
from monocycle_nash.domain.character import Character, MatchupVector
from monocycle_nash.domain.matrix.approximation import (
    DominantEigenpairMonocycleApproximation,
    EquilibriumPreservingResidualMonocycleApproximation,
    MonocycleToGeneralApproximation,
)
from monocycle_nash.domain.matrix.base import PayoffMatrix
from monocycle_nash.domain.matrix.builder import PayoffMatrixBuilder
from monocycle_nash.domain.team import Team
from monocycle_nash.domain.visualization.character_vector_graph import CharacterVectorGraphPlotter
from monocycle_nash.domain.visualization.payoff_graph import PayoffDirectedGraphPlotter


# ---------------------------------------------------------------------------
# NodeResolutionContext — ノードが使用するコンテキストインターフェース
# ---------------------------------------------------------------------------


class NodeResolutionContext(ABC):
    """MatrixNode.build() が再帰解決やファイル読み込みに使用するコンテキスト。"""

    @abstractmethod
    def resolve_node(self, node: MatrixNode) -> PayoffMatrix:
        """別ノードを再帰的に解決して PayoffMatrix を返す。"""
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
# Character nodes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CharacterNode:
    """インラインのキャラクター設定ノード。"""

    power: float
    vector: tuple[float, float]
    label: str = ""


class CharacterSource(ABC):
    """キャラクター入力ソースの抽象基底。"""

    @abstractmethod
    def load_characters(self, ctx: NodeResolutionContext) -> list[Character]:
        """キャラクターリストを返す。"""
        raise NotImplementedError


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


class TeamSource(ABC):
    """チーム入力ソースの抽象基底。"""

    @abstractmethod
    def load_teams(self, ctx: NodeResolutionContext) -> list[Team]:
        """チームリストを返す。"""
        raise NotImplementedError


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


class OutputNode(ABC):
    """出力ノードの抽象基底。"""

    @abstractmethod
    def run(
        self,
        *,
        output_path_port: OutputPathPort,
        node_name: str,
        matrix: PayoffMatrix,
    ) -> Path:
        """出力を実行してファイルパスを返す。"""
        raise NotImplementedError


@dataclass(frozen=True)
class PayoffDirectedGraphOutputNode(OutputNode):
    """有向グラフ出力設定ノード。"""

    filename: str = "payoff_directed_graph.svg"
    threshold: float = 0.0
    canvas_size: int = 840

    def run(
        self,
        *,
        output_path_port: OutputPathPort,
        node_name: str,
        matrix: PayoffMatrix,
    ) -> Path:
        path = output_path_port.resolve_output_path(
            node_name=node_name,
            output_method="payoff_directed_graph",
            filename=self.filename,
        )
        PayoffDirectedGraphPlotter(
            payoff_matrix=matrix.matrix,
            labels=matrix.labels,
            threshold=self.threshold,
        ).draw(path, canvas_size=self.canvas_size)
        return path


@dataclass(frozen=True)
class CharacterVectorGraphOutputNode(OutputNode):
    """キャラクターベクトルグラフ出力設定ノード。"""

    filename: str = "character_vector_graph.svg"
    canvas_size: int = 840
    margin: int = 90

    def run(
        self,
        *,
        output_path_port: OutputPathPort,
        node_name: str,
        matrix: PayoffMatrix,
    ) -> Path:
        path = output_path_port.resolve_output_path(
            node_name=node_name,
            output_method="character_vector_graph",
            filename=self.filename,
        )
        characters = getattr(matrix, "characters", None)
        if not isinstance(characters, list) or not characters:
            raise ValueError(
                "character_vector_graph は characters を持つノードでのみ使用できます"
            )
        CharacterVectorGraphPlotter(characters).draw(
            output_path=path,
            canvas_size=self.canvas_size,
            margin=self.margin,
        )
        return path


# ---------------------------------------------------------------------------
# MatrixNode — 行列構築ノードの抽象基底
# ---------------------------------------------------------------------------


class MatrixNode(ABC):
    """行列構築ノードの抽象基底。

    すべての具象ノードは name・outputs フィールドと build メソッドを実装する。
    """

    name: str
    outputs: tuple[OutputNode, ...]

    @abstractmethod
    def build(self, ctx: NodeResolutionContext) -> PayoffMatrix:
        """コンテキストを使って PayoffMatrix を構築して返す。"""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Matrix nodes — 各生成方式ごとの具象ノード
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GeneralFromRawNode(MatrixNode):
    """生行列データから一般利得行列を構築するノード。"""

    matrix: Any  # list[list[float]] または np.ndarray
    labels: list[str] | None = None
    name: str = "root"
    outputs: tuple[OutputNode, ...] = field(default_factory=tuple)

    def build(self, ctx: NodeResolutionContext) -> PayoffMatrix:
        matrix = np.asarray(self.matrix, dtype=float)
        return PayoffMatrixBuilder.from_general_matrix(matrix=matrix, labels=self.labels)


@dataclass(frozen=True)
class MonocycleFromCharactersNode(MatrixNode):
    """キャラクターリストから単相性モデル利得行列を構築するノード。

    characters には CharacterInlineSource または CharacterListFromFileNode を指定する。
    """

    characters: CharacterSource
    labels: list[str] | None = None
    name: str = "root"
    outputs: tuple[OutputNode, ...] = field(default_factory=tuple)

    def build(self, ctx: NodeResolutionContext) -> PayoffMatrix:
        characters = self.characters.load_characters(ctx)
        return PayoffMatrixBuilder.from_characters(characters=characters, labels=self.labels)


@dataclass(frozen=True)
class GeneralFromTeamsPayoffNode(MatrixNode):
    """計算済みチーム利得行列から一般利得行列を構築するノード。

    teams には TeamInlineSource または TeamListFromFileNode を指定する。
    """

    team_payoff: Any  # list[list[float]] または np.ndarray
    teams: TeamSource
    name: str = "root"
    outputs: tuple[OutputNode, ...] = field(default_factory=tuple)

    def build(self, ctx: NodeResolutionContext) -> PayoffMatrix:
        team_payoff = np.asarray(self.team_payoff, dtype=float)
        teams = self.teams.load_teams(ctx)
        return PayoffMatrixBuilder.from_teams(team_payoff=team_payoff, teams=teams)


@dataclass(frozen=True)
class GeneralFromTeamMatchupsNode(MatrixNode):
    """キャラクター行列とチーム定義からチーム利得行列を構築するノード。

    character_matrix には任意の MatrixNode を再帰的に指定できる。
    teams には TeamInlineSource または TeamListFromFileNode を指定する。
    """

    teams: TeamSource
    character_matrix: MatrixNode
    use_monocycle_formula: bool = True
    name: str = "root"
    outputs: tuple[OutputNode, ...] = field(default_factory=tuple)

    def build(self, ctx: NodeResolutionContext) -> PayoffMatrix:
        character_matrix = ctx.resolve_node(self.character_matrix)
        teams = self.teams.load_teams(ctx)
        return PayoffMatrixBuilder.from_team_matchups(
            teams=teams,
            character_matrix=character_matrix,
            use_monocycle_formula=self.use_monocycle_formula,
        )


@dataclass(frozen=True)
class RandomSkewSymmetricNode(MatrixNode):
    """ランダム交代行列を生成するノード。"""

    size: int
    low: float = -1.0
    high: float = 1.0
    seed: int | None = None
    max_attempts: int = 10_000
    labels: list[str] | None = None
    name: str = "root"
    outputs: tuple[OutputNode, ...] = field(default_factory=tuple)

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
class ApproxMonocycleToGeneralNode(MatrixNode):
    """単相性行列を一般行列へ変換する近似ノード。

    source には任意の MatrixNode を再帰的に指定できる。
    """

    source: MatrixNode
    name: str = "root"
    outputs: tuple[OutputNode, ...] = field(default_factory=tuple)

    def build(self, ctx: NodeResolutionContext) -> PayoffMatrix:
        source = ctx.resolve_node(self.source)
        return MonocycleToGeneralApproximation().approximate(source).matrix


@dataclass(frozen=True)
class ApproxDominantEigenpairNode(MatrixNode):
    """支配固有値ペアによる近似変換ノード。

    source には任意の MatrixNode を再帰的に指定できる。
    """

    source: MatrixNode
    atol: float = 1e-8
    name: str = "root"
    outputs: tuple[OutputNode, ...] = field(default_factory=tuple)

    def build(self, ctx: NodeResolutionContext) -> PayoffMatrix:
        source = ctx.resolve_node(self.source)
        return DominantEigenpairMonocycleApproximation(atol=self.atol).approximate(source).matrix


@dataclass(frozen=True)
class ApproxEquilibriumPreservingNode(MatrixNode):
    """均衡保存残差近似変換ノード。

    source には任意の MatrixNode を再帰的に指定できる。
    """

    source: MatrixNode
    atol: float = 1e-8
    name: str = "root"
    outputs: tuple[OutputNode, ...] = field(default_factory=tuple)

    def build(self, ctx: NodeResolutionContext) -> PayoffMatrix:
        source = ctx.resolve_node(self.source)
        return EquilibriumPreservingResidualMonocycleApproximation(atol=self.atol).approximate(source).matrix
