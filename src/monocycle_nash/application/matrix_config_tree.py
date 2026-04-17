"""Payoff matrix node resolution for application layer orchestration.

設定ツリーを型付きノードで構成し、MatrixConfigTreeResolver が
_ResolutionSession を生成して解決を委譲する。
各解決ロジックは各ノードの build / run / load_characters / load_teams に実装されている。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from monocycle_nash.application.matrix_nodes import (
    MatrixNode,
    NodeResolutionContext,
    OutputNode,
)
from monocycle_nash.application.ports import (
    CharacterListFilePort,
    MatrixFilePort,
    OutputPathPort,
    TeamListFilePort,
)
from monocycle_nash.domain.character import Character
from monocycle_nash.domain.matrix.base import PayoffMatrix
from monocycle_nash.domain.team import Team


@dataclass(frozen=True)
class MatrixConfigTree:
    """利得行列構築設定ツリー。"""

    root: MatrixNode


@dataclass(frozen=True)
class ResolvedOutput:
    """1つの出力実行結果。"""

    node_name: str
    output_node: OutputNode
    path: Path


@dataclass(frozen=True)
class MatrixResolutionResult:
    """設定ツリー解決結果。"""

    root: PayoffMatrix
    outputs: tuple[ResolvedOutput, ...] = ()


class MatrixConfigTreeResolver:
    """設定ツリーを辿って最終的なオブジェクト生成・出力実行を行う。

    ファイル読み込みが必要なノードを解決するためのポートを
    コンストラクタで受け取り、resolve 呼び出しごとに _ResolutionSession を生成する。
    """

    def __init__(
        self,
        *,
        output_path_port: OutputPathPort | None = None,
        character_list_file_port: CharacterListFilePort | None = None,
        team_list_file_port: TeamListFilePort | None = None,
        matrix_file_port: MatrixFilePort | None = None,
    ):
        self._output_path_port = output_path_port
        self._character_list_file_port = character_list_file_port
        self._team_list_file_port = team_list_file_port
        self._matrix_file_port = matrix_file_port

    def resolve(self, tree: MatrixConfigTree) -> MatrixResolutionResult:
        session = _ResolutionSession(
            output_path_port=self._output_path_port,
            character_list_file_port=self._character_list_file_port,
            team_list_file_port=self._team_list_file_port,
        )
        root = session.resolve_node(tree.root)
        return MatrixResolutionResult(root=root, outputs=tuple(session.resolved_outputs))


class _ResolutionSession(NodeResolutionContext):
    """単一の resolve() 呼び出しに対応するセッション。

    NodeResolutionContext を実装し、各ノードの build から呼び出される。
    キャッシュと出力結果リストを保持する。
    """

    def __init__(
        self,
        *,
        output_path_port: OutputPathPort | None,
        character_list_file_port: CharacterListFilePort | None,
        team_list_file_port: TeamListFilePort | None,
    ) -> None:
        self._output_path_port = output_path_port
        self._character_list_file_port = character_list_file_port
        self._team_list_file_port = team_list_file_port
        self.resolved_outputs: list[ResolvedOutput] = []
        self._resolved_cache: dict[int, PayoffMatrix] = {}

    def resolve_node(self, node: MatrixNode) -> PayoffMatrix:
        cache_key = id(node)
        if cache_key in self._resolved_cache:
            return self._resolved_cache[cache_key]

        resolved = node.build(self)
        self._resolved_cache[cache_key] = resolved

        for output_node in node.outputs:
            if self._output_path_port is None:
                raise ValueError("出力を実行するには OutputPathPort が必要です")
            path = output_node.run(
                output_path_port=self._output_path_port,
                node_name=node.name,
                matrix=resolved,
            )
            self.resolved_outputs.append(
                ResolvedOutput(node_name=node.name, output_node=output_node, path=path)
            )

        return resolved

    def load_characters_from_file(self, path: str) -> list[Character]:
        if self._character_list_file_port is None:
            raise ValueError(
                "CharacterListFromFileNode を解決するには CharacterListFilePort が必要です"
            )
        return self._character_list_file_port.load_characters(path)

    def load_teams_from_file(self, path: str) -> list[Team]:
        if self._team_list_file_port is None:
            raise ValueError(
                "TeamListFromFileNode を解決するには TeamListFilePort が必要です"
            )
        return self._team_list_file_port.load_teams(path)

