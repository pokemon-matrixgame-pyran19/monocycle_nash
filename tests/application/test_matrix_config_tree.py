from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import pytest

from monocycle_nash.application.matrix_config_tree import (
    MatrixBuildMethod,
    MatrixConfigNode,
    MatrixConfigTree,
    MatrixConfigTreeFactory,
    MatrixConfigTreeResolver,
    OutputConfigNode,
    OutputMethod,
)
from monocycle_nash.application.ports import ConfigReferencePort, OutputPathPort
from monocycle_nash.domain.matrix.general import GeneralPayoffMatrix
from monocycle_nash.domain.matrix.monocycle import MonocyclePayoffMatrix


class StubConfigReferencePort(ConfigReferencePort):
    def __init__(self, mapping: Mapping[str, Mapping[str, Any]]):
        self._mapping = dict(mapping)

    def load_config(self, reference: str) -> Mapping[str, Any]:
        return dict(self._mapping[reference])


class StubOutputPathPort(OutputPathPort):
    def __init__(self, base_dir: Path):
        self._base_dir = base_dir

    def resolve_output_path(
        self,
        *,
        node_name: str,
        output_method: str,
        filename: str,
    ) -> Path:
        return self._base_dir / f"{node_name}-{output_method}-{filename}"


def test_factory_build_tree_with_reference_and_dependency() -> None:
    reference_port = StubConfigReferencePort(
        {
            "character-source": {
                "method": "monocycle_from_characters",
                "settings": {
                    "characters": [
                        {"label": "A", "p": 1.0, "v": [1.0, 0.0]},
                        {"label": "B", "p": 0.5, "v": [0.0, 1.0]},
                    ]
                },
            }
        }
    )
    factory = MatrixConfigTreeFactory(reference_port=reference_port)
    tree = factory.build_tree(
        {
            "method": "approx_monocycle_to_general",
            "dependencies": {
                "source": {
                    "$ref": "character-source",
                }
            },
            "outputs": [
                {
                    "method": "payoff_directed_graph",
                    "settings": {"threshold": 0.2, "filename": "root.svg"},
                }
            ],
        }
    )

    assert tree.root.method == MatrixBuildMethod.APPROX_MONOCYCLE_TO_GENERAL
    assert "source" in tree.root.dependencies
    assert tree.root.dependencies["source"].method == MatrixBuildMethod.MONOCYCLE_FROM_CHARACTERS
    assert tree.root.outputs[0].method == OutputMethod.PAYOFF_DIRECTED_GRAPH


def test_resolver_builds_team_matrix_with_nested_dependency_and_outputs(tmp_path: Path) -> None:
    tree = MatrixConfigTree(
        root=MatrixConfigNode(
            name="team-root",
            method=MatrixBuildMethod.GENERAL_FROM_TEAM_MATCHUPS,
            settings={
                "teams": [
                    {"label": "A+B", "members": ["A", "B"]},
                    {"label": "B+C", "members": ["B", "C"]},
                ],
                "use_monocycle_formula": True,
            },
            dependencies={
                "character_matrix": MatrixConfigNode(
                    name="character-source",
                    method=MatrixBuildMethod.MONOCYCLE_FROM_CHARACTERS,
                    settings={
                        "characters": [
                            {"label": "A", "p": 1.0, "v": [1.0, 0.0]},
                            {"label": "B", "p": 0.0, "v": [0.0, 1.0]},
                            {"label": "C", "p": -1.0, "v": [-1.0, 0.0]},
                        ],
                        "labels": ["A", "B", "C"],
                    },
                    outputs=(
                        OutputConfigNode(
                            method=OutputMethod.CHARACTER_VECTOR_GRAPH,
                            settings={"filename": "chars.svg"},
                        ),
                    ),
                )
            },
            outputs=(
                OutputConfigNode(
                    method=OutputMethod.PAYOFF_DIRECTED_GRAPH,
                    settings={"filename": "team.svg"},
                ),
            ),
        )
    )

    resolver = MatrixConfigTreeResolver(output_path_port=StubOutputPathPort(tmp_path))
    result = resolver.resolve(tree)

    assert isinstance(result.root, GeneralPayoffMatrix)
    assert result.root.matrix.shape == (2, 2)
    assert len(result.outputs) == 2
    assert all(output.path.exists() for output in result.outputs)


def test_resolver_without_output_path_port_rejects_output() -> None:
    tree = MatrixConfigTree(
        root=MatrixConfigNode(
            name="root",
            method=MatrixBuildMethod.MONOCYCLE_FROM_CHARACTERS,
            settings={
                "characters": [
                    {"label": "A", "p": 1.0, "v": [1.0, 0.0]},
                    {"label": "B", "p": 0.0, "v": [0.0, 1.0]},
                ]
            },
            outputs=(
                OutputConfigNode(
                    method=OutputMethod.CHARACTER_VECTOR_GRAPH,
                    settings={"filename": "chars.svg"},
                ),
            ),
        )
    )
    resolver = MatrixConfigTreeResolver()

    with pytest.raises(ValueError, match="OutputPathPort"):
        resolver.resolve(tree)


def test_static_factory_monocycle_tree_resolves_to_monocycle_matrix() -> None:
    tree = MatrixConfigTreeFactory.monocycle_from_characters(
        characters=[
            {"label": "A", "p": 1.0, "v": [1.0, 0.0]},
            {"label": "B", "p": 0.2, "v": [0.0, 1.0]},
        ],
        labels=["A", "B"],
    )

    result = MatrixConfigTreeResolver().resolve(tree)
    assert isinstance(result.root, MonocyclePayoffMatrix)
    assert result.root.labels == ["A", "B"]
