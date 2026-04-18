"""NodeSpec から typed MatrixNode を組み立てるファクトリ。

インフラ層がロードした NodeSpec を受け取り、
対応する具象 MatrixNode インスタンスに変換して返す。

ノード生成ロジックをアプリ層に集約することで、
インフラ層（TOML 等）はデータ構造の変換だけに専念できる。

新規ノード種別を追加する場合は:
  1. matrix_nodes.py に具象 MatrixNode を追加する
  2. MatrixNodeFactory に対応する `_build_<method>` プライベートメソッドを追加する
  3. __init__ の _builders 辞書にエントリを追加する
  build() 自体は変更不要。
"""

from __future__ import annotations

from typing import Callable

from monocycle_nash.application.matrix_nodes import (
    ApproxDominantEigenpairNode,
    ApproxEquilibriumPreservingNode,
    ApproxMonocycleToGeneralNode,
    CharacterInlineSource,
    CharacterListFromFileNode,
    CharacterNode,
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
    TeamInlineSource,
    TeamListFromFileNode,
    TeamNode,
    TeamSource,
)
from monocycle_nash.application.node_spec import NodeSpec, OutputSpec


class MatrixNodeFactory:
    """NodeSpec から typed MatrixNode を組み立てるファクトリ。

    method 文字列をキーとするビルダー辞書にディスパッチすることで、
    build() メソッド自体を変更せずに新規ノード種別を追加できる。
    refs に補助データへのパスが含まれる場合はファイル参照ノードを生成し、
    params にインラインデータが含まれる場合はインラインソースを生成する。
    """

    def __init__(self) -> None:
        self._builders: dict[str, Callable[[NodeSpec], MatrixNode]] = {
            "general_from_raw": self._build_general_from_raw,
            "monocycle_from_characters": self._build_monocycle_from_characters,
            "general_from_teams_payoff": self._build_general_from_teams_payoff,
            "general_from_team_matchups": self._build_general_from_team_matchups,
            "random_skew_symmetric": self._build_random_skew_symmetric,
            "approx_monocycle_to_general": self._build_approx_monocycle_to_general,
            "approx_dominant_eigenpair": self._build_approx_dominant_eigenpair,
            "approx_equilibrium_preserving": self._build_approx_equilibrium_preserving,
        }
        self._output_builders: dict[str, Callable[[OutputSpec], OutputNode]] = {
            "payoff_directed_graph": self._build_payoff_directed_graph_output,
            "character_vector_graph": self._build_character_vector_graph_output,
        }

    def build(self, spec: NodeSpec) -> MatrixNode:
        """NodeSpec を対応する MatrixNode に変換して返す。"""
        builder = self._builders.get(spec.method)
        if builder is None:
            raise ValueError(f"未知の method です: {spec.method!r}")
        return builder(spec)

    # ------------------------------------------------------------------
    # MatrixNode ビルダー — method ごとに1つのプライベートメソッド
    # ------------------------------------------------------------------

    def _build_general_from_raw(self, spec: NodeSpec) -> MatrixNode:
        return GeneralFromRawNode(
            matrix=spec.params["matrix"],
            labels=spec.params.get("labels"),
            name=spec.name,
            outputs=self._build_outputs(spec.outputs),
        )

    def _build_monocycle_from_characters(self, spec: NodeSpec) -> MatrixNode:
        return MonocycleFromCharactersNode(
            characters=self._build_character_source(spec),
            labels=spec.params.get("labels"),
            name=spec.name,
            outputs=self._build_outputs(spec.outputs),
        )

    def _build_general_from_teams_payoff(self, spec: NodeSpec) -> MatrixNode:
        return GeneralFromTeamsPayoffNode(
            team_payoff=spec.params["team_payoff"],
            teams=self._build_team_source(spec),
            name=spec.name,
            outputs=self._build_outputs(spec.outputs),
        )

    def _build_general_from_team_matchups(self, spec: NodeSpec) -> MatrixNode:
        character_matrix_spec = spec.children.get("character_matrix")
        if character_matrix_spec is None:
            raise ValueError(
                "general_from_team_matchups には children.character_matrix が必要です"
            )
        return GeneralFromTeamMatchupsNode(
            teams=self._build_team_source(spec),
            character_matrix=self.build(character_matrix_spec),
            use_monocycle_formula=spec.params.get("use_monocycle_formula", True),
            name=spec.name,
            outputs=self._build_outputs(spec.outputs),
        )

    def _build_random_skew_symmetric(self, spec: NodeSpec) -> MatrixNode:
        return RandomSkewSymmetricNode(
            size=spec.params["size"],
            low=spec.params.get("low", -1.0),
            high=spec.params.get("high", 1.0),
            seed=spec.params.get("seed"),
            max_attempts=spec.params.get("max_attempts", 10_000),
            labels=spec.params.get("labels"),
            name=spec.name,
            outputs=self._build_outputs(spec.outputs),
        )

    def _build_approx_monocycle_to_general(self, spec: NodeSpec) -> MatrixNode:
        source_spec = spec.children.get("source")
        if source_spec is None:
            raise ValueError("approx_monocycle_to_general には children.source が必要です")
        return ApproxMonocycleToGeneralNode(
            source=self.build(source_spec),
            name=spec.name,
            outputs=self._build_outputs(spec.outputs),
        )

    def _build_approx_dominant_eigenpair(self, spec: NodeSpec) -> MatrixNode:
        source_spec = spec.children.get("source")
        if source_spec is None:
            raise ValueError("approx_dominant_eigenpair には children.source が必要です")
        return ApproxDominantEigenpairNode(
            source=self.build(source_spec),
            atol=spec.params.get("atol", 1e-8),
            name=spec.name,
            outputs=self._build_outputs(spec.outputs),
        )

    def _build_approx_equilibrium_preserving(self, spec: NodeSpec) -> MatrixNode:
        source_spec = spec.children.get("source")
        if source_spec is None:
            raise ValueError("approx_equilibrium_preserving には children.source が必要です")
        return ApproxEquilibriumPreservingNode(
            source=self.build(source_spec),
            atol=spec.params.get("atol", 1e-8),
            name=spec.name,
            outputs=self._build_outputs(spec.outputs),
        )

    # ------------------------------------------------------------------
    # ソース・出力ビルダー
    # ------------------------------------------------------------------

    def _build_character_source(self, spec: NodeSpec) -> CharacterSource:
        """refs または params.characters からキャラクターソースを生成する。"""
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

    def _build_team_source(self, spec: NodeSpec) -> TeamSource:
        """refs または params.teams からチームソースを生成する。"""
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

    def _build_outputs(self, output_specs: tuple[OutputSpec, ...]) -> tuple[OutputNode, ...]:
        """OutputSpec のタプルを OutputNode のタプルに変換する。"""
        return tuple(self._build_output(s) for s in output_specs)

    def _build_output(self, spec: OutputSpec) -> OutputNode:
        """OutputSpec を対応する OutputNode に変換して返す。"""
        builder = self._output_builders.get(spec.method)
        if builder is None:
            raise ValueError(f"未知の output method です: {spec.method!r}")
        return builder(spec)

    def _build_payoff_directed_graph_output(self, spec: OutputSpec) -> OutputNode:
        return PayoffDirectedGraphOutputNode(
            filename=spec.params.get("filename", "payoff_directed_graph.svg"),
            threshold=spec.params.get("threshold", 0.0),
            canvas_size=spec.params.get("canvas_size", 840),
        )

    def _build_character_vector_graph_output(self, spec: OutputSpec) -> OutputNode:
        return CharacterVectorGraphOutputNode(
            filename=spec.params.get("filename", "character_vector_graph.svg"),
            canvas_size=spec.params.get("canvas_size", 840),
            margin=spec.params.get("margin", 90),
        )
