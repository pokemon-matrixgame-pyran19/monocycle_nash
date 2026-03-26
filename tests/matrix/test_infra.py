from __future__ import annotations

from pathlib import Path

from monocycle_nash.matrix.infra import MatrixFileInfrastructure


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8")


def test_matrix_file_infrastructure_loads_matrix(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    _write(data_dir / "matrix" / "rps3" / "data.toml", 'matrix = [[0, 1], [-1, 0]]')

    infra = MatrixFileInfrastructure(base_dir=data_dir)
    matrix = infra.load_matrix("rps3")

    assert matrix.matrix.tolist() == [[0.0, 1.0], [-1.0, 0.0]]


def test_matrix_file_infrastructure_initializes_payoff_matrix(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    _write(
        data_dir / "matrix" / "character_model" / "data.toml",
        '''
        [[characters]]
        label = "rock"
        p = 1.0
        v = [1.0, 0.0]

        [[characters]]
        label = "paper"
        p = 1.0
        v = [0.0, 1.0]
        ''',
    )

    infra = MatrixFileInfrastructure(base_dir=data_dir)
    matrix = infra.load_matrix("character_model")

    assert matrix.matrix.shape == (2, 2)
    assert matrix.labels == ["rock", "paper"]
