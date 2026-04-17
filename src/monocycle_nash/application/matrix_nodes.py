"""Application-layer typed node definitions for payoff matrix construction.

各ノード型が「この生成方式はこういう値や設定を受け取る」を明示する。
依存する他ドメインモデルは型付きフィールドとして直接保持し、
末端でファイル読み込みが必要なノードは対応するポートで解決する。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Union


# ---------------------------------------------------------------------------
# Character nodes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CharacterNode:
    """インラインのキャラクター設定ノード。"""

    power: float
    vector: tuple[float, float]
    label: str = ""


@dataclass(frozen=True)
class CharacterListFromFileNode:
    """ファイルからキャラクターリストを読み込む設定ノード。

    インフラ層の CharacterListFilePort を呼び出して解決する。
    """

    path: str


CharacterSource = Union[list[CharacterNode], CharacterListFromFileNode]


# ---------------------------------------------------------------------------
# Team nodes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TeamNode:
    """インラインのチーム設定ノード。"""

    label: str
    member_ids: tuple[str | int, ...]


@dataclass(frozen=True)
class TeamListFromFileNode:
    """ファイルからチームリストを読み込む設定ノード。

    インフラ層の TeamListFilePort を呼び出して解決する。
    """

    path: str


TeamSource = Union[list[TeamNode], TeamListFromFileNode]


# ---------------------------------------------------------------------------
# Output nodes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PayoffDirectedGraphOutputNode:
    """有向グラフ出力設定ノード。"""

    filename: str = "payoff_directed_graph.svg"
    threshold: float = 0.0
    canvas_size: int = 840


@dataclass(frozen=True)
class CharacterVectorGraphOutputNode:
    """キャラクターベクトルグラフ出力設定ノード。"""

    filename: str = "character_vector_graph.svg"
    canvas_size: int = 840
    margin: int = 90


OutputNode = Union[PayoffDirectedGraphOutputNode, CharacterVectorGraphOutputNode]


# ---------------------------------------------------------------------------
# Matrix nodes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GeneralFromRawNode:
    """生行列データから一般利得行列を構築するノード。"""

    matrix: Any  # list[list[float]] または np.ndarray
    labels: list[str] | None = None
    name: str = "root"
    outputs: tuple[OutputNode, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class MonocycleFromCharactersNode:
    """キャラクターリストから単相性モデル利得行列を構築するノード。

    characters には CharacterNode のリスト、または CharacterListFromFileNode を指定する。
    """

    characters: CharacterSource
    labels: list[str] | None = None
    name: str = "root"
    outputs: tuple[OutputNode, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class GeneralFromTeamsPayoffNode:
    """計算済みチーム利得行列から一般利得行列を構築するノード。

    teams には TeamNode のリスト、または TeamListFromFileNode を指定する。
    """

    team_payoff: Any  # list[list[float]] または np.ndarray
    teams: TeamSource
    name: str = "root"
    outputs: tuple[OutputNode, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class GeneralFromTeamMatchupsNode:
    """キャラクター行列とチーム定義からチーム利得行列を構築するノード。

    character_matrix には任意の MatrixNode を再帰的に指定できる。
    teams には TeamNode のリスト、または TeamListFromFileNode を指定する。
    """

    teams: TeamSource
    character_matrix: "MatrixNode"
    use_monocycle_formula: bool = True
    name: str = "root"
    outputs: tuple[OutputNode, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class RandomSkewSymmetricNode:
    """ランダム交代行列を生成するノード。"""

    size: int
    low: float = -1.0
    high: float = 1.0
    seed: int | None = None
    max_attempts: int = 10_000
    labels: list[str] | None = None
    name: str = "root"
    outputs: tuple[OutputNode, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class ApproxMonocycleToGeneralNode:
    """単相性行列を一般行列へ変換する近似ノード。

    source には任意の MatrixNode を再帰的に指定できる。
    """

    source: "MatrixNode"
    name: str = "root"
    outputs: tuple[OutputNode, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class ApproxDominantEigenpairNode:
    """支配固有値ペアによる近似変換ノード。

    source には任意の MatrixNode を再帰的に指定できる。
    """

    source: "MatrixNode"
    atol: float = 1e-8
    name: str = "root"
    outputs: tuple[OutputNode, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class ApproxEquilibriumPreservingNode:
    """均衡保存残差近似変換ノード。

    source には任意の MatrixNode を再帰的に指定できる。
    """

    source: "MatrixNode"
    atol: float = 1e-8
    name: str = "root"
    outputs: tuple[OutputNode, ...] = field(default_factory=tuple)


MatrixNode = Union[
    GeneralFromRawNode,
    MonocycleFromCharactersNode,
    GeneralFromTeamsPayoffNode,
    GeneralFromTeamMatchupsNode,
    RandomSkewSymmetricNode,
    ApproxMonocycleToGeneralNode,
    ApproxDominantEigenpairNode,
    ApproxEquilibriumPreservingNode,
]
