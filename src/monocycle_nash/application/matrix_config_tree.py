"""Payoff matrix node resolution for application layer orchestration.

設定ツリーを型付きノードで構成し、MatrixConfigTreeResolver が
各ノード型に対応した解決ロジックをインフラ層ポート経由で実行する。
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import singledispatchmethod
from pathlib import Path

import numpy as np

from monocycle_nash.application.matrix_nodes import (
    ApproxDominantEigenpairNode,
    ApproxEquilibriumPreservingNode,
    ApproxMonocycleToGeneralNode,
    CharacterListFromFileNode,
    CharacterVectorGraphOutputNode,
    GeneralFromRawNode,
    GeneralFromTeamMatchupsNode,
    GeneralFromTeamsPayoffNode,
    MatrixNode,
    MonocycleFromCharactersNode,
    OutputNode,
    PayoffDirectedGraphOutputNode,
    RandomSkewSymmetricNode,
    TeamListFromFileNode,
)
from monocycle_nash.application.ports import (
    CharacterListFilePort,
    MatrixFilePort,
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
from monocycle_nash.domain.team import Team
from monocycle_nash.domain.visualization.character_vector_graph import CharacterVectorGraphPlotter
from monocycle_nash.domain.visualization.payoff_graph import PayoffDirectedGraphPlotter


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
    コンストラクタで受け取り、各 resolve メソッドで明示的に呼び出す。
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
        outputs: list[ResolvedOutput] = []
        resolved_cache: dict[int, PayoffMatrix] = {}
        root = self._resolve_node(tree.root, outputs=outputs, resolved_cache=resolved_cache)
        return MatrixResolutionResult(root=root, outputs=tuple(outputs))

    # ------------------------------------------------------------------
    # Matrix node resolution
    # ------------------------------------------------------------------

    def _resolve_node(
        self,
        node: MatrixNode,
        *,
        outputs: list[ResolvedOutput],
        resolved_cache: dict[int, PayoffMatrix],
    ) -> PayoffMatrix:
        cache_key = id(node)
        if cache_key in resolved_cache:
            return resolved_cache[cache_key]

        resolved = self._build_matrix(node, outputs=outputs, resolved_cache=resolved_cache)
        resolved_cache[cache_key] = resolved

        for output_node in node.outputs:
            path = self._run_output(output_node, node_name=node.name, matrix=resolved)
            outputs.append(ResolvedOutput(node_name=node.name, output_node=output_node, path=path))

        return resolved

    @singledispatchmethod
    def _build_matrix(
        self,
        node: object,
        *,
        outputs: list[ResolvedOutput],
        resolved_cache: dict[int, PayoffMatrix],
    ) -> PayoffMatrix:
        raise TypeError(f"未対応のノード型: {type(node)}")

    @_build_matrix.register
    def _(
        self,
        node: GeneralFromRawNode,
        *,
        outputs: list[ResolvedOutput],
        resolved_cache: dict[int, PayoffMatrix],
    ) -> PayoffMatrix:
        matrix = np.asarray(node.matrix, dtype=float)
        return PayoffMatrixBuilder.from_general_matrix(matrix=matrix, labels=node.labels)

    @_build_matrix.register
    def _(
        self,
        node: MonocycleFromCharactersNode,
        *,
        outputs: list[ResolvedOutput],
        resolved_cache: dict[int, PayoffMatrix],
    ) -> PayoffMatrix:
        characters = self._resolve_characters(node.characters)
        return PayoffMatrixBuilder.from_characters(characters=characters, labels=node.labels)

    @_build_matrix.register
    def _(
        self,
        node: GeneralFromTeamsPayoffNode,
        *,
        outputs: list[ResolvedOutput],
        resolved_cache: dict[int, PayoffMatrix],
    ) -> PayoffMatrix:
        team_payoff = np.asarray(node.team_payoff, dtype=float)
        teams = self._resolve_teams(node.teams)
        return PayoffMatrixBuilder.from_teams(team_payoff=team_payoff, teams=teams)

    @_build_matrix.register
    def _(
        self,
        node: GeneralFromTeamMatchupsNode,
        *,
        outputs: list[ResolvedOutput],
        resolved_cache: dict[int, PayoffMatrix],
    ) -> PayoffMatrix:
        character_matrix = self._resolve_node(
            node.character_matrix, outputs=outputs, resolved_cache=resolved_cache
        )
        teams = self._resolve_teams(node.teams)
        return PayoffMatrixBuilder.from_team_matchups(
            teams=teams,
            character_matrix=character_matrix,
            use_monocycle_formula=node.use_monocycle_formula,
        )

    @_build_matrix.register
    def _(
        self,
        node: RandomSkewSymmetricNode,
        *,
        outputs: list[ResolvedOutput],
        resolved_cache: dict[int, PayoffMatrix],
    ) -> PayoffMatrix:
        rng = np.random.default_rng(node.seed) if node.seed is not None else None
        return PayoffMatrixBuilder.from_random_matrix(
            size=node.size,
            low=node.low,
            high=node.high,
            rng=rng,
            max_attempts=node.max_attempts,
            labels=node.labels,
        )

    @_build_matrix.register
    def _(
        self,
        node: ApproxMonocycleToGeneralNode,
        *,
        outputs: list[ResolvedOutput],
        resolved_cache: dict[int, PayoffMatrix],
    ) -> PayoffMatrix:
        source = self._resolve_node(node.source, outputs=outputs, resolved_cache=resolved_cache)
        return MonocycleToGeneralApproximation().approximate(source).matrix

    @_build_matrix.register
    def _(
        self,
        node: ApproxDominantEigenpairNode,
        *,
        outputs: list[ResolvedOutput],
        resolved_cache: dict[int, PayoffMatrix],
    ) -> PayoffMatrix:
        source = self._resolve_node(node.source, outputs=outputs, resolved_cache=resolved_cache)
        return DominantEigenpairMonocycleApproximation(atol=node.atol).approximate(source).matrix

    @_build_matrix.register
    def _(
        self,
        node: ApproxEquilibriumPreservingNode,
        *,
        outputs: list[ResolvedOutput],
        resolved_cache: dict[int, PayoffMatrix],
    ) -> PayoffMatrix:
        source = self._resolve_node(node.source, outputs=outputs, resolved_cache=resolved_cache)
        return EquilibriumPreservingResidualMonocycleApproximation(atol=node.atol).approximate(source).matrix

    # ------------------------------------------------------------------
    # Character / Team resolution (file-backed or inline)
    # ------------------------------------------------------------------

    @singledispatchmethod
    def _resolve_characters(self, source: object) -> list[Character]:
        raise TypeError(f"未対応のキャラクターソース型: {type(source)}")

    @_resolve_characters.register
    def _(self, source: CharacterListFromFileNode) -> list[Character]:
        if self._character_list_file_port is None:
            raise ValueError(
                "CharacterListFromFileNode を解決するには CharacterListFilePort が必要です"
            )
        return self._character_list_file_port.load_characters(source.path)

    @_resolve_characters.register
    def _(self, source: list) -> list[Character]:
        return [
            Character(c.power, MatchupVector(c.vector[0], c.vector[1]), c.label)
            for c in source
        ]

    @singledispatchmethod
    def _resolve_teams(self, source: object) -> list[Team]:
        raise TypeError(f"未対応のチームソース型: {type(source)}")

    @_resolve_teams.register
    def _(self, source: TeamListFromFileNode) -> list[Team]:
        if self._team_list_file_port is None:
            raise ValueError(
                "TeamListFromFileNode を解決するには TeamListFilePort が必要です"
            )
        return self._team_list_file_port.load_teams(source.path)

    @_resolve_teams.register
    def _(self, source: list) -> list[Team]:
        return [Team(label=t.label, member_ids=t.member_ids) for t in source]

    # ------------------------------------------------------------------
    # Output execution
    # ------------------------------------------------------------------

    @singledispatchmethod
    def _run_output(
        self,
        output_node: object,
        *,
        node_name: str,
        matrix: PayoffMatrix,
    ) -> Path:
        raise TypeError(f"未対応の出力ノード型: {type(output_node)}")

    @_run_output.register
    def _(
        self,
        output_node: PayoffDirectedGraphOutputNode,
        *,
        node_name: str,
        matrix: PayoffMatrix,
    ) -> Path:
        if self._output_path_port is None:
            raise ValueError("出力を実行するには OutputPathPort が必要です")

        path = self._output_path_port.resolve_output_path(
            node_name=node_name,
            output_method="payoff_directed_graph",
            filename=output_node.filename,
        )
        PayoffDirectedGraphPlotter(
            payoff_matrix=matrix.matrix,
            labels=matrix.labels,
            threshold=output_node.threshold,
        ).draw(path, canvas_size=output_node.canvas_size)
        return path

    @_run_output.register
    def _(
        self,
        output_node: CharacterVectorGraphOutputNode,
        *,
        node_name: str,
        matrix: PayoffMatrix,
    ) -> Path:
        if self._output_path_port is None:
            raise ValueError("出力を実行するには OutputPathPort が必要です")

        path = self._output_path_port.resolve_output_path(
            node_name=node_name,
            output_method="character_vector_graph",
            filename=output_node.filename,
        )
        characters = getattr(matrix, "characters", None)
        if not isinstance(characters, list) or not characters:
            raise ValueError(
                "character_vector_graph は characters を持つノードでのみ使用できます"
            )
        CharacterVectorGraphPlotter(characters).draw(
            output_path=path,
            canvas_size=output_node.canvas_size,
            margin=output_node.margin,
        )
        return path
