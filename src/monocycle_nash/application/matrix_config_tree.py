"""Payoff matrix node resolution for application layer orchestration.

設定ツリーを型付きノードで構成し、MatrixConfigTreeResolver が
_ResolutionSession を生成して解決を委譲する。
各解決ロジックは各ノードの build / run / load_characters / load_teams に実装されている。
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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

    run_id: int
    root: PayoffMatrix
    outputs: tuple[ResolvedOutput, ...] = ()


class MatrixConfigTreeResolver:
    """設定ツリーを辿って最終的なオブジェクト生成・出力実行を行う。

    ファイル読み込みが必要なノードを解決するためのポートを
    コンストラクタで受け取り、resolve 呼び出しごとに _ResolutionSession を生成する。
    resolve を呼ぶたびに連番の run_id を振る。
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
        self._run_counter = itertools.count(1)

    def resolve(self, tree: MatrixConfigTree) -> MatrixResolutionResult:
        run_id = next(self._run_counter)
        session = _ResolutionSession(
            run_id=run_id,
            output_path_port=self._output_path_port,
            character_list_file_port=self._character_list_file_port,
            team_list_file_port=self._team_list_file_port,
        )
        root = session.resolve_node(tree.root)
        return MatrixResolutionResult(
            run_id=run_id,
            root=root,
            outputs=tuple(session.resolved_outputs),
        )


class _ResolutionSession(NodeResolutionContext):
    """単一の resolve() 呼び出しに対応するセッション。

    NodeResolutionContext を実装し、各ノードの build から呼び出される。
    キャッシュと出力結果リストを保持する。
    """

    def __init__(
        self,
        *,
        run_id: int,
        output_path_port: OutputPathPort | None,
        character_list_file_port: CharacterListFilePort | None,
        team_list_file_port: TeamListFilePort | None,
    ) -> None:
        self.run_id = run_id
        self._output_path_port = output_path_port
        self._character_list_file_port = character_list_file_port
        self._team_list_file_port = team_list_file_port
        self.resolved_outputs: list[ResolvedOutput] = []
        self._resolved_cache: dict[int, PayoffMatrix] = {}
        self._active_node_path_stack: list[tuple[str, ...]] = []
        self._trace_records_by_path: dict[tuple[str, ...], dict[str, Any]] = {}

    def resolve_node(self, node: MatrixNode) -> PayoffMatrix:
        cache_key = id(node)
        if self._active_node_path_stack:
            node_path = (*self._active_node_path_stack[-1], node.name)
        else:
            node_path = (node.name,)
        if cache_key in self._resolved_cache:
            # 同一ノード参照は初回探索時に1回だけ解決し、出力実行も初回のみ行う。
            return self._resolved_cache[cache_key]

        self._active_node_path_stack.append(node_path)
        self._trace_records_by_path[node_path] = self._build_trace_record(node, node_path)
        try:
            resolved = node.build(self)
        finally:
            self._active_node_path_stack.pop()
        self._resolved_cache[cache_key] = resolved
        self._trace_records_by_path[node_path] = self._build_trace_record(
            node,
            node_path,
            resolved,
        )

        for output_node in node.outputs:
            if self._output_path_port is None:
                raise ValueError("出力を実行するには OutputPathPort が必要です")
            path = output_node.run(
                output_path_port=self._output_path_port,
                run_id=str(self.run_id),
                node_path=node_path,
                matrix=resolved,
                trace_records=self._collect_trace_records(node_path),
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

    def _collect_trace_records(self, node_path: tuple[str, ...]) -> tuple[dict[str, Any], ...]:
        records: list[dict[str, Any]] = []
        for i in range(1, len(node_path) + 1):
            path_prefix = node_path[:i]
            record = self._trace_records_by_path.get(path_prefix)
            if record is not None:
                records.append(dict(record))
        return tuple(records)

    def _build_trace_record(
        self,
        node: MatrixNode,
        node_path: tuple[str, ...],
        resolved: PayoffMatrix | None = None,
    ) -> dict[str, Any]:
        intermediate = (
            node.output_intermediate_values(resolved)
            if resolved is not None
            else {}
        )
        return {
            "node_path": list(node_path),
            "node_name": node.name,
            "node_method": node.node_method_name,
            "normalized_params": node.normalized_trace_params(),
            "intermediate_values": intermediate,
        }
