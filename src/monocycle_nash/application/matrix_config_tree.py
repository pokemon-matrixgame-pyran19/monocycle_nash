"""Payoff matrix node resolution for application layer orchestration.

設定ツリーを型付きノードで構成し、MatrixConfigTreeResolver が
各ノード型に対応した解決ロジックをインフラ層ポート経由で実行する。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from monocycle_nash.application.matrix_nodes import (
    ApproxDominantEigenpairNode,
    ApproxEquilibriumPreservingNode,
    ApproxMonocycleToGeneralNode,
    CharacterListFromFileNode,
    CharacterSource,
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
    TeamSource,
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
            path = self._run_output(node_name=node.name, matrix=resolved, output_node=output_node)
            outputs.append(ResolvedOutput(node_name=node.name, output_node=output_node, path=path))

        return resolved

    def _build_matrix(
        self,
        node: MatrixNode,
        *,
        outputs: list[ResolvedOutput],
        resolved_cache: dict[int, PayoffMatrix],
    ) -> PayoffMatrix:
        if isinstance(node, GeneralFromRawNode):
            matrix = np.asarray(node.matrix, dtype=float)
            return PayoffMatrixBuilder.from_general_matrix(matrix=matrix, labels=node.labels)

        if isinstance(node, MonocycleFromCharactersNode):
            characters = self._resolve_characters(node.characters)
            return PayoffMatrixBuilder.from_characters(characters=characters, labels=node.labels)

        if isinstance(node, GeneralFromTeamsPayoffNode):
            team_payoff = np.asarray(node.team_payoff, dtype=float)
            teams = self._resolve_teams(node.teams)
            return PayoffMatrixBuilder.from_teams(team_payoff=team_payoff, teams=teams)

        if isinstance(node, GeneralFromTeamMatchupsNode):
            character_matrix = self._resolve_node(
                node.character_matrix, outputs=outputs, resolved_cache=resolved_cache
            )
            teams = self._resolve_teams(node.teams)
            return PayoffMatrixBuilder.from_team_matchups(
                teams=teams,
                character_matrix=character_matrix,
                use_monocycle_formula=node.use_monocycle_formula,
            )

        if isinstance(node, RandomSkewSymmetricNode):
            rng = np.random.default_rng(node.seed) if node.seed is not None else None
            return PayoffMatrixBuilder.from_random_matrix(
                size=node.size,
                low=node.low,
                high=node.high,
                rng=rng,
                max_attempts=node.max_attempts,
                labels=node.labels,
            )

        if isinstance(node, ApproxMonocycleToGeneralNode):
            source = self._resolve_node(node.source, outputs=outputs, resolved_cache=resolved_cache)
            return MonocycleToGeneralApproximation().approximate(source).matrix

        if isinstance(node, ApproxDominantEigenpairNode):
            source = self._resolve_node(node.source, outputs=outputs, resolved_cache=resolved_cache)
            return DominantEigenpairMonocycleApproximation(atol=node.atol).approximate(source).matrix

        if isinstance(node, ApproxEquilibriumPreservingNode):
            source = self._resolve_node(node.source, outputs=outputs, resolved_cache=resolved_cache)
            return EquilibriumPreservingResidualMonocycleApproximation(atol=node.atol).approximate(source).matrix

        raise TypeError(f"未対応のノード型: {type(node)}")

    # ------------------------------------------------------------------
    # Character / Team resolution (file-backed or inline)
    # ------------------------------------------------------------------

    def _resolve_characters(self, source: CharacterSource) -> list[Character]:
        if isinstance(source, CharacterListFromFileNode):
            if self._character_list_file_port is None:
                raise ValueError(
                    "CharacterListFromFileNode を解決するには CharacterListFilePort が必要です"
                )
            return self._character_list_file_port.load_characters(source.path)
        return [
            Character(c.power, MatchupVector(c.vector[0], c.vector[1]), c.label)
            for c in source
        ]

    def _resolve_teams(self, source: TeamSource) -> list[Team]:
        if isinstance(source, TeamListFromFileNode):
            if self._team_list_file_port is None:
                raise ValueError(
                    "TeamListFromFileNode を解決するには TeamListFilePort が必要です"
                )
            return self._team_list_file_port.load_teams(source.path)
        return [Team(label=t.label, member_ids=t.member_ids) for t in source]

    # ------------------------------------------------------------------
    # Output execution
    # ------------------------------------------------------------------

    def _run_output(
        self,
        *,
        node_name: str,
        matrix: PayoffMatrix,
        output_node: OutputNode,
    ) -> Path:
        if self._output_path_port is None:
            raise ValueError("出力を実行するには OutputPathPort が必要です")

        if isinstance(output_node, PayoffDirectedGraphOutputNode):
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

        if isinstance(output_node, CharacterVectorGraphOutputNode):
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

        raise TypeError(f"未対応の出力ノード型: {type(output_node)}")

