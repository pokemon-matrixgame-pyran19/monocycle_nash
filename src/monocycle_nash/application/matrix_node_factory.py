"""NodeSpec から typed MatrixNode を組み立てるファクトリ。

インフラ層がロードした NodeSpec を受け取り、
対応する具象 MatrixNode インスタンスに変換して返す。

ノード生成ロジックをアプリ層に集約することで、
インフラ層（TOML 等）はデータ構造の変換だけに専念できる。
"""

from __future__ import annotations

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

    NodeSpec.method の値に応じて適切な MatrixNode サブクラスを生成する。
    refs に補助データへのパスが含まれる場合はファイル参照ノードを生成し、
    params にインラインデータが含まれる場合はインラインソースを生成する。
    """

    def build(self, spec: NodeSpec) -> MatrixNode:
        """NodeSpec を対応する MatrixNode に変換して返す。"""
        outputs = self._build_outputs(spec.outputs)
        method = spec.method

        if method == "general_from_raw":
            return GeneralFromRawNode(
                matrix=spec.params["matrix"],
                labels=spec.params.get("labels"),
                name=spec.name,
                outputs=outputs,
            )

        if method == "monocycle_from_characters":
            return MonocycleFromCharactersNode(
                characters=self._build_character_source(spec),
                labels=spec.params.get("labels"),
                name=spec.name,
                outputs=outputs,
            )

        if method == "general_from_teams_payoff":
            return GeneralFromTeamsPayoffNode(
                team_payoff=spec.params["team_payoff"],
                teams=self._build_team_source(spec),
                name=spec.name,
                outputs=outputs,
            )

        if method == "general_from_team_matchups":
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
                outputs=outputs,
            )

        if method == "random_skew_symmetric":
            return RandomSkewSymmetricNode(
                size=spec.params["size"],
                low=spec.params.get("low", -1.0),
                high=spec.params.get("high", 1.0),
                seed=spec.params.get("seed"),
                max_attempts=spec.params.get("max_attempts", 10_000),
                labels=spec.params.get("labels"),
                name=spec.name,
                outputs=outputs,
            )

        if method == "approx_monocycle_to_general":
            source_spec = spec.children.get("source")
            if source_spec is None:
                raise ValueError(
                    "approx_monocycle_to_general には children.source が必要です"
                )
            return ApproxMonocycleToGeneralNode(
                source=self.build(source_spec),
                name=spec.name,
                outputs=outputs,
            )

        if method == "approx_dominant_eigenpair":
            source_spec = spec.children.get("source")
            if source_spec is None:
                raise ValueError(
                    "approx_dominant_eigenpair には children.source が必要です"
                )
            return ApproxDominantEigenpairNode(
                source=self.build(source_spec),
                atol=spec.params.get("atol", 1e-8),
                name=spec.name,
                outputs=outputs,
            )

        if method == "approx_equilibrium_preserving":
            source_spec = spec.children.get("source")
            if source_spec is None:
                raise ValueError(
                    "approx_equilibrium_preserving には children.source が必要です"
                )
            return ApproxEquilibriumPreservingNode(
                source=self.build(source_spec),
                atol=spec.params.get("atol", 1e-8),
                name=spec.name,
                outputs=outputs,
            )

        raise ValueError(f"未知の method です: {method!r}")

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
        if spec.method == "payoff_directed_graph":
            return PayoffDirectedGraphOutputNode(
                filename=spec.params.get("filename", "payoff_directed_graph.svg"),
                threshold=spec.params.get("threshold", 0.0),
                canvas_size=spec.params.get("canvas_size", 840),
            )
        if spec.method == "character_vector_graph":
            return CharacterVectorGraphOutputNode(
                filename=spec.params.get("filename", "character_vector_graph.svg"),
                canvas_size=spec.params.get("canvas_size", 840),
                margin=spec.params.get("margin", 90),
            )
        raise ValueError(f"未知の output method です: {spec.method!r}")
