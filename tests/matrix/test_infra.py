from __future__ import annotations

from pathlib import Path

from monocycle_nash.infrastructure.input.matrix_reader import FileMatrixDataReader
from monocycle_nash.application.matrix_construction import MatrixConstructionUseCase
from monocycle_nash.application.dto import MatrixInputDTO


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8")


def test_file_matrix_data_reader_loads_raw(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    _write(data_dir / "matrix" / "rps3" / "data.toml", 'matrix = [[0, 1], [-1, 0]]')

    reader = FileMatrixDataReader(data_dir=data_dir)
    data = reader.load_matrix_data("rps3")

    assert data["matrix"] == [[0, 1], [-1, 0]]


def test_file_matrix_data_reader_builds_matrix(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    _write(data_dir / "matrix" / "rps3" / "data.toml", 'matrix = [[0, 1], [-1, 0]]')

    reader = FileMatrixDataReader(data_dir=data_dir)
    data = reader.load_matrix_data("rps3")

    dto = MatrixInputDTO(raw_matrix=data.get("matrix"), labels=data.get("labels"))
    uc = MatrixConstructionUseCase()
    matrix, _ = uc.build(dto)

    assert matrix.matrix.tolist() == [[0.0, 1.0], [-1.0, 0.0]]


def test_file_matrix_data_reader_characters(tmp_path: Path) -> None:
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

    reader = FileMatrixDataReader(data_dir=data_dir)
    data = reader.load_matrix_data("character_model")

    dto = MatrixInputDTO(characters=data.get("characters"))
    uc = MatrixConstructionUseCase()
    matrix, _ = uc.build(dto)

    assert matrix.matrix.shape == (2, 2)
    assert matrix.labels == ["rock", "paper"]
