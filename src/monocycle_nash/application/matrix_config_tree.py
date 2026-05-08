"""Payoff matrix node resolution for application layer orchestration.

設定ツリーを型付きノードで構成し、MatrixConfigTreeResolver が
_ResolutionSession を生成して解決を委譲する。
各解決ロジックは各ノードの build / emit / execute / load_characters / load_teams に実装されている。
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from pathlib import Path

from monocycle_nash.application.matrix_nodes import (
    MatrixNode,
    NodeResolutionContext,
    OutputEmission,
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
    runner: str
    path: Path


@dataclass(frozen=True)
class ResolvedOutputEmission:
    """1つの出力送信イベント。"""

    node_name: str
    output_node: OutputNode
    runner: str
    node_path: tuple[str, ...]


@dataclass(frozen=True)
class ResolvedRunner:
    """Runner の最終実行結果。"""

    runner: str
    emitted_count: int
    output_count: int


@dataclass(frozen=True)
class MatrixResolutionResult:
    """設定ツリー解決結果。"""

    run_id: int
    root: PayoffMatrix
    output_emissions: tuple[ResolvedOutputEmission, ...] = ()
    runners: tuple[ResolvedRunner, ...] = ()
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
        resolved_outputs, resolved_runners = session.run_output_runners()
        return MatrixResolutionResult(
            run_id=run_id,
            root=root,
            output_emissions=tuple(session.resolved_output_emissions),
            runners=tuple(resolved_runners),
            outputs=tuple(resolved_outputs),
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
        self.resolved_output_emissions: list[ResolvedOutputEmission] = []
        self._emissions_by_runner: dict[str, list[OutputEmission]] = {}
        self._resolved_cache: dict[int, PayoffMatrix] = {}
        self._active_node_path_stack: list[tuple[str, ...]] = []

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
        try:
            resolved = node.build(self)
        finally:
            self._active_node_path_stack.pop()
        self._resolved_cache[cache_key] = resolved

        domains = node.provide_domains(resolved)
        for output_index, output_node in enumerate(node.outputs):
            runner = output_node.resolve_runner() or self._build_default_runner_id(
                node_path=node_path,
                output_method=output_node.output_method,
                output_index=output_index,
            )
            emission = output_node.emit(
                node_name=node.name,
                node_path=node_path,
                domains=domains,
            )
            self._emissions_by_runner.setdefault(runner, []).append(emission)
            self.resolved_output_emissions.append(
                ResolvedOutputEmission(
                    node_name=node.name,
                    output_node=output_node,
                    runner=runner,
                    node_path=node_path,
                )
            )

        return resolved

    def run_output_runners(self) -> tuple[tuple[ResolvedOutput, ...], tuple[ResolvedRunner, ...]]:
        if self._emissions_by_runner and self._output_path_port is None:
            raise ValueError("出力を実行するには OutputPathPort が必要です")
        if self._output_path_port is None:
            return (), ()
        output_path_port = self._output_path_port

        resolved_outputs: list[ResolvedOutput] = []
        resolved_runners: list[ResolvedRunner] = []
        for runner, emissions in self._emissions_by_runner.items():
            output_count = 0
            for emission in emissions:
                path = emission.output_node.execute(
                    output_path_port=output_path_port,
                    run_id=str(self.run_id),
                    node_path=emission.node_path,
                    domains=emission.domains,
                )
                output_count += 1
                resolved_outputs.append(
                    ResolvedOutput(
                        node_name=emission.node_name,
                        output_node=emission.output_node,
                        runner=runner,
                        path=path,
                    )
                )
            resolved_runners.append(
                ResolvedRunner(
                    runner=runner,
                    emitted_count=len(emissions),
                    output_count=output_count,
                )
            )
        return tuple(resolved_outputs), tuple(resolved_runners)

    @staticmethod
    def _build_default_runner_id(
        *,
        node_path: tuple[str, ...],
        output_method: str,
        output_index: int,
    ) -> str:
        return "__single__:" + "/".join(node_path) + f":{output_method}:{output_index}"

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
