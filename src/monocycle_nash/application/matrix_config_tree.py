"""Payoff matrix configuration tree for application layer orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from monocycle_nash.application.ports import ConfigReferencePort, OutputPathPort
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


class MatrixBuildMethod(StrEnum):
    """利得行列（または行列変換結果）の構築方式。"""

    GENERAL_FROM_RAW = "general_from_raw"
    MONOCYCLE_FROM_CHARACTERS = "monocycle_from_characters"
    GENERAL_FROM_TEAMS_PAYOFF = "general_from_teams_payoff"
    GENERAL_FROM_TEAM_MATCHUPS = "general_from_team_matchups"
    RANDOM_SKEW_SYMMETRIC = "random_skew_symmetric"
    APPROX_MONOCYCLE_TO_GENERAL = "approx_monocycle_to_general"
    APPROX_DOMINANT_EIGENPAIR = "approx_dominant_eigenpair"
    APPROX_EQUILIBRIUM_PRESERVING = "approx_equilibrium_preserving"


class OutputMethod(StrEnum):
    """ノード出力方式。"""

    PAYOFF_DIRECTED_GRAPH = "payoff_directed_graph"
    CHARACTER_VECTOR_GRAPH = "character_vector_graph"


@dataclass(frozen=True)
class OutputConfigNode:
    """ツリー上ノードの出力設定。"""

    method: OutputMethod
    settings: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MatrixConfigNode:
    """利得行列構築の設定ノード。"""

    name: str
    method: MatrixBuildMethod
    settings: Mapping[str, Any] = field(default_factory=dict)
    dependencies: Mapping[str, "MatrixConfigNode"] = field(default_factory=dict)
    outputs: tuple[OutputConfigNode, ...] = ()


@dataclass(frozen=True)
class MatrixConfigTree:
    """利得行列構築設定ツリー。"""

    root: MatrixConfigNode


@dataclass(frozen=True)
class ResolvedOutput:
    """1つの出力実行結果。"""

    node_name: str
    method: OutputMethod
    path: Path


@dataclass(frozen=True)
class MatrixResolutionResult:
    """設定ツリー解決結果。"""

    root: PayoffMatrix
    outputs: tuple[ResolvedOutput, ...] = ()


class MatrixConfigTreeFactory:
    """外部入力（dict等）から設定ツリーを生成する。"""

    def __init__(self, *, reference_port: ConfigReferencePort | None = None):
        self._reference_port = reference_port

    def build_tree(self, raw_config: Mapping[str, Any]) -> MatrixConfigTree:
        node = self._parse_node(raw_config, node_name="root")
        return MatrixConfigTree(root=node)

    @staticmethod
    def general_from_raw(
        matrix: list[list[float]] | np.ndarray,
        *,
        labels: list[str] | None = None,
        name: str = "root",
        outputs: tuple[OutputConfigNode, ...] = (),
    ) -> MatrixConfigTree:
        return MatrixConfigTree(
            root=MatrixConfigNode(
                name=name,
                method=MatrixBuildMethod.GENERAL_FROM_RAW,
                settings={"matrix": matrix, "labels": labels},
                outputs=outputs,
            )
        )

    @staticmethod
    def monocycle_from_characters(
        characters: list[Character] | list[Mapping[str, Any]],
        *,
        labels: list[str] | None = None,
        name: str = "root",
        outputs: tuple[OutputConfigNode, ...] = (),
    ) -> MatrixConfigTree:
        return MatrixConfigTree(
            root=MatrixConfigNode(
                name=name,
                method=MatrixBuildMethod.MONOCYCLE_FROM_CHARACTERS,
                settings={"characters": characters, "labels": labels},
                outputs=outputs,
            )
        )

    def _materialize_reference(self, raw: Mapping[str, Any]) -> dict[str, Any]:
        payload = dict(raw)
        ref = payload.pop("$ref", None)
        if ref is None:
            return payload
        if self._reference_port is None:
            raise ValueError("`$ref` を使うには ConfigReferencePort が必要です")

        loaded = dict(self._reference_port.load_config(str(ref)))
        loaded.update(payload)
        return loaded

    def _parse_node(self, raw: Mapping[str, Any], *, node_name: str) -> MatrixConfigNode:
        payload = self._materialize_reference(raw)

        if "method" not in payload:
            raise ValueError(f"method が必要です: node={node_name}")
        method = MatrixBuildMethod(payload["method"])
        settings = payload.get("settings", {})
        if not isinstance(settings, Mapping):
            raise ValueError(f"settings はマッピングである必要があります: node={node_name}")

        dependency_payload = payload.get("dependencies", {})
        if not isinstance(dependency_payload, Mapping):
            raise ValueError(f"dependencies はマッピングである必要があります: node={node_name}")
        dependencies: dict[str, MatrixConfigNode] = {}
        for dep_name, dep_raw in dependency_payload.items():
            if not isinstance(dep_raw, Mapping):
                raise ValueError(
                    f"dependency `{dep_name}` はノード設定(dict)である必要があります: node={node_name}"
                )
            dependencies[str(dep_name)] = self._parse_node(
                dep_raw,
                node_name=f"{node_name}.{dep_name}",
            )

        outputs: list[OutputConfigNode] = []
        output_payload = payload.get("outputs", [])
        if not isinstance(output_payload, list):
            raise ValueError(f"outputs は配列である必要があります: node={node_name}")
        for item in output_payload:
            if not isinstance(item, Mapping):
                raise ValueError(f"output 要素はマッピングである必要があります: node={node_name}")
            if "method" not in item:
                raise ValueError(f"output.method が必要です: node={node_name}")
            output_settings = item.get("settings", {})
            if not isinstance(output_settings, Mapping):
                raise ValueError(f"output.settings はマッピングである必要があります: node={node_name}")
            outputs.append(OutputConfigNode(method=OutputMethod(item["method"]), settings=output_settings))

        explicit_name = payload.get("name")
        final_name = str(explicit_name) if explicit_name else node_name
        return MatrixConfigNode(
            name=final_name,
            method=method,
            settings=dict(settings),
            dependencies=dependencies,
            outputs=tuple(outputs),
        )


class MatrixConfigTreeResolver:
    """設定ツリーを辿って最終的なオブジェクト生成・出力実行を行う。"""

    def __init__(self, *, output_path_port: OutputPathPort | None = None):
        self._output_path_port = output_path_port

    def resolve(self, tree: MatrixConfigTree) -> MatrixResolutionResult:
        outputs: list[ResolvedOutput] = []
        resolved_cache: dict[int, PayoffMatrix] = {}
        root = self._resolve_node(tree.root, outputs=outputs, resolved_cache=resolved_cache)
        return MatrixResolutionResult(root=root, outputs=tuple(outputs))

    def _resolve_node(
        self,
        node: MatrixConfigNode,
        *,
        outputs: list[ResolvedOutput],
        resolved_cache: dict[int, PayoffMatrix],
    ) -> PayoffMatrix:
        cache_key = id(node)
        if cache_key in resolved_cache:
            return resolved_cache[cache_key]

        dependencies = {
            key: self._resolve_node(child, outputs=outputs, resolved_cache=resolved_cache)
            for key, child in node.dependencies.items()
        }
        resolved = self._build_matrix(node, dependencies=dependencies)
        resolved_cache[cache_key] = resolved

        for output in node.outputs:
            output_path = self._run_output(node=node, matrix=resolved, output=output)
            outputs.append(
                ResolvedOutput(node_name=node.name, method=output.method, path=output_path)
            )
        return resolved

    def _build_matrix(
        self,
        node: MatrixConfigNode,
        *,
        dependencies: Mapping[str, PayoffMatrix],
    ) -> PayoffMatrix:
        settings = node.settings

        if node.method == MatrixBuildMethod.GENERAL_FROM_RAW:
            matrix = np.asarray(settings["matrix"], dtype=float)
            labels = settings.get("labels")
            return PayoffMatrixBuilder.from_general_matrix(matrix=matrix, labels=labels)

        if node.method == MatrixBuildMethod.MONOCYCLE_FROM_CHARACTERS:
            labels = settings.get("labels")
            characters = self._coerce_characters(settings["characters"])
            return PayoffMatrixBuilder.from_characters(characters=characters, labels=labels)

        if node.method == MatrixBuildMethod.GENERAL_FROM_TEAMS_PAYOFF:
            team_payoff = np.asarray(settings["team_payoff"], dtype=float)
            teams = self._coerce_teams(settings["teams"])
            return PayoffMatrixBuilder.from_teams(team_payoff=team_payoff, teams=teams)

        if node.method == MatrixBuildMethod.GENERAL_FROM_TEAM_MATCHUPS:
            character_matrix = dependencies.get("character_matrix")
            if character_matrix is None:
                raise ValueError("GENERAL_FROM_TEAM_MATCHUPS には dependency `character_matrix` が必要です")
            teams = self._coerce_teams(settings["teams"])
            use_monocycle_formula = bool(settings.get("use_monocycle_formula", True))
            return PayoffMatrixBuilder.from_team_matchups(
                teams=teams,
                character_matrix=character_matrix,
                use_monocycle_formula=use_monocycle_formula,
            )

        if node.method == MatrixBuildMethod.RANDOM_SKEW_SYMMETRIC:
            seed = settings.get("seed")
            rng = np.random.default_rng(int(seed)) if seed is not None else None
            return PayoffMatrixBuilder.from_random_matrix(
                size=int(settings["size"]),
                low=float(settings.get("low", -1.0)),
                high=float(settings.get("high", 1.0)),
                rng=rng,
                max_attempts=int(settings.get("max_attempts", 10_000)),
                labels=settings.get("labels"),
            )

        if node.method == MatrixBuildMethod.APPROX_MONOCYCLE_TO_GENERAL:
            source = self._required_dependency(dependencies, key="source")
            result = MonocycleToGeneralApproximation().approximate(source)
            return result.matrix

        if node.method == MatrixBuildMethod.APPROX_DOMINANT_EIGENPAIR:
            source = self._required_dependency(dependencies, key="source")
            approximation = DominantEigenpairMonocycleApproximation(
                atol=float(settings.get("atol", 1e-8))
            )
            result = approximation.approximate(source)
            return result.matrix

        if node.method == MatrixBuildMethod.APPROX_EQUILIBRIUM_PRESERVING:
            source = self._required_dependency(dependencies, key="source")
            approximation = EquilibriumPreservingResidualMonocycleApproximation(
                atol=float(settings.get("atol", 1e-8))
            )
            result = approximation.approximate(source)
            return result.matrix

        raise ValueError(f"未対応の method: {node.method}")

    @staticmethod
    def _required_dependency(dependencies: Mapping[str, PayoffMatrix], *, key: str) -> PayoffMatrix:
        dependency = dependencies.get(key)
        if dependency is None:
            raise ValueError(f"dependency `{key}` が必要です")
        return dependency

    @staticmethod
    def _coerce_characters(raw: Any) -> list[Character]:
        if not isinstance(raw, list) or not raw:
            raise ValueError("characters は1件以上の配列で指定してください")

        characters: list[Character] = []
        for i, item in enumerate(raw):
            if isinstance(item, Character):
                characters.append(item)
                continue
            if not isinstance(item, Mapping):
                raise ValueError(f"characters[{i}] は Character または dict で指定してください")
            label = str(item.get("label", ""))
            power = float(item["p"])
            vector_raw = item["v"]
            vector = MatchupVector(float(vector_raw[0]), float(vector_raw[1]))
            characters.append(Character(power=power, vector=vector, label=label))
        return characters

    @staticmethod
    def _coerce_teams(raw: Any) -> list[Team]:
        if not isinstance(raw, list) or not raw:
            raise ValueError("teams は1件以上の配列で指定してください")

        teams: list[Team] = []
        for i, item in enumerate(raw):
            if isinstance(item, Team):
                teams.append(item)
                continue
            if not isinstance(item, Mapping):
                raise ValueError(f"teams[{i}] は Team または dict で指定してください")
            members = item.get("members")
            if not isinstance(members, list) or not members:
                raise ValueError(f"teams[{i}].members は1件以上の配列で指定してください")
            teams.append(
                Team(
                    label=str(item["label"]),
                    member_ids=tuple(members),
                )
            )
        return teams

    def _run_output(
        self,
        *,
        node: MatrixConfigNode,
        matrix: PayoffMatrix,
        output: OutputConfigNode,
    ) -> Path:
        if self._output_path_port is None:
            raise ValueError("出力を実行するには OutputPathPort が必要です")

        filename = str(output.settings.get("filename", f"{output.method}.svg"))
        path = self._output_path_port.resolve_output_path(
            node_name=node.name,
            output_method=str(output.method),
            filename=filename,
        )

        if output.method == OutputMethod.PAYOFF_DIRECTED_GRAPH:
            threshold = float(output.settings.get("threshold", 0.0))
            canvas_size = int(output.settings.get("canvas_size", 840))
            PayoffDirectedGraphPlotter(
                payoff_matrix=matrix.matrix,
                labels=matrix.labels,
                threshold=threshold,
            ).draw(path, canvas_size=canvas_size)
            return path

        if output.method == OutputMethod.CHARACTER_VECTOR_GRAPH:
            characters = getattr(matrix, "characters", None)
            if not isinstance(characters, list) or not characters:
                raise ValueError(
                    "character_vector_graph は characters を持つノードでのみ使用できます"
                )
            canvas_size = int(output.settings.get("canvas_size", 840))
            margin = int(output.settings.get("margin", 90))
            CharacterVectorGraphPlotter(characters).draw(
                output_path=path,
                canvas_size=canvas_size,
                margin=margin,
            )
            return path

        raise ValueError(f"未対応の output method: {output.method}")
