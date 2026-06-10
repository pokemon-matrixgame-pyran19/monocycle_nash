import json
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import MagicMock

from monocycle_nash.application.matrix_nodes import (
    PayoffMatrixOutputNode,
    MatrixNode,
    NodeResolutionContext,
)
from monocycle_nash.application.node_spec import OutputSpec
from monocycle_nash.application.ports import OutputPathPort
from monocycle_nash.domain.matrix.base import PayoffMatrix
from monocycle_nash.domain.strategy import PureStrategySet


class MockPayoffMatrix(PayoffMatrix):
    def __init__(self, matrix, row_labels, col_labels):
        self._matrix = np.asarray(matrix)
        self._row_strategies = PureStrategySet.from_labels(row_labels)
        self._col_strategies = PureStrategySet.from_labels(col_labels)

    @property
    def matrix(self) -> np.ndarray:
        return self._matrix

    @property
    def size(self) -> int:
        return self._matrix.shape[0]

    @property
    def row_strategies(self) -> PureStrategySet:
        return self._row_strategies

    @property
    def col_strategies(self) -> PureStrategySet:
        return self._col_strategies


def test_payoff_matrix_output_full(tmp_path: Path):
    # Setup
    matrix_data = [[0.0, 1.0], [-1.0, 0.0]]
    row_labels = ["R1", "R2"]
    col_labels = ["C1", "C2"]
    mock_matrix = MockPayoffMatrix(matrix_data, row_labels, col_labels)

    node = MagicMock(spec=MatrixNode)
    node.provide_object.return_value = mock_matrix

    ctx = MagicMock(spec=NodeResolutionContext)

    port = MagicMock(spec=OutputPathPort)
    output_file = tmp_path / "output.json"
    port.resolve_output_path.return_value = output_file

    spec = OutputSpec(method="payoff_matrix", params={"filename": "output.json"})
    output_node = PayoffMatrixOutputNode.create_from_spec(spec)

    # Execute
    res_path = output_node.execute(
        output_path_port=port,
        run_id="run1",
        node_path=("root",),
        node=node,
        ctx=ctx
    )

    # Verify
    assert res_path == output_file
    with open(output_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["matrix"] == matrix_data
    assert data["row_labels"] == row_labels
    assert data["col_labels"] == col_labels


def test_payoff_matrix_output_sliced(tmp_path: Path):
    # Setup
    matrix_data = [
        [0.0, 1.0, 2.0],
        [-1.0, 0.0, 1.0],
        [-2.0, -1.0, 0.0]
    ]
    row_labels = ["R1", "R2", "R3"]
    col_labels = ["C1", "C2", "C3"]
    mock_matrix = MockPayoffMatrix(matrix_data, row_labels, col_labels)

    node = MagicMock(spec=MatrixNode)
    node.provide_object.return_value = mock_matrix

    ctx = MagicMock(spec=NodeResolutionContext)

    port = MagicMock(spec=OutputPathPort)
    output_file = tmp_path / "sliced.json"
    port.resolve_output_path.return_value = output_file

    spec = OutputSpec(
        method="payoff_matrix",
        params={
            "filename": "sliced.json",
            "rows": [0, 2],
            "cols": [1]
        }
    )
    output_node = PayoffMatrixOutputNode.create_from_spec(spec)

    # Execute
    output_node.execute(
        output_path_port=port,
        run_id="run1",
        node_path=("root",),
        node=node,
        ctx=ctx
    )

    # Verify
    with open(output_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # row 0,2 and col 1
    # [0.0, 1.0, 2.0] -> index 1 is 1.0
    # [-2.0, -1.0, 0.0] -> index 1 is -1.0
    assert data["matrix"] == [[1.0], [-1.0]]
    assert data["row_labels"] == ["R1", "R3"]
    assert data["col_labels"] == ["C2"]
