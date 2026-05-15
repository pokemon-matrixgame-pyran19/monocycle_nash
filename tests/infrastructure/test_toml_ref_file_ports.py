from __future__ import annotations

from pathlib import Path

import pytest

from monocycle_nash.infrastructure.input import (
    TomlCharacterListFilePort,
    TomlTeamListFilePort,
)


def test_load_characters_from_relative_ref_with_suffix_completion(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    chars_path = data_dir / "characters"
    chars_path.mkdir()
    (chars_path / "rps.toml").write_text(
        """
[[characters]]
power = 1.0
vector = [1.0, 0.0]
label = "Rock"

[[characters]]
power = 0.0
vector = [0.0, 1.0]
label = "Paper"
""".strip(),
        encoding="utf-8",
    )

    port = TomlCharacterListFilePort(data_dir=data_dir)
    characters = port.load_characters("characters/rps")

    assert len(characters) == 2
    assert characters[0].label == "Rock"
    assert characters[1].label == "Paper"


def test_load_characters_from_directory_data_toml(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    source_dir = data_dir / "characters" / "set1"
    source_dir.mkdir(parents=True)
    (source_dir / "data.toml").write_text(
        """
[[characters]]
power = 1.0
vector = [0.5, -0.5]
""".strip(),
        encoding="utf-8",
    )

    port = TomlCharacterListFilePort(data_dir=data_dir)
    characters = port.load_characters("characters/set1")

    assert len(characters) == 1
    assert characters[0].p == pytest.approx(1.0)


def test_load_characters_from_csv_ref(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    chars_path = data_dir / "characters"
    chars_path.mkdir(parents=True)
    (chars_path / "rps.csv").write_text(
        """
label,power,vector_x,vector_y
Rock,1.0,1.0,0.0
Paper,0.0,0.0,1.0
""".strip(),
        encoding="utf-8",
    )

    port = TomlCharacterListFilePort(data_dir=data_dir)
    characters = port.load_characters("characters/rps.csv")

    assert len(characters) == 2
    assert characters[0].label == "Rock"
    assert characters[1].label == "Paper"


def test_load_characters_from_csv_with_missing_required_columns_raises(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    chars_path = data_dir / "characters"
    chars_path.mkdir(parents=True)
    (chars_path / "invalid.csv").write_text(
        """
label,power,vector_x
Rock,1.0,1.0
""".strip(),
        encoding="utf-8",
    )

    port = TomlCharacterListFilePort(data_dir=data_dir)
    with pytest.raises(ValueError, match="必須列がありません"):
        port.load_characters("characters/invalid.csv")


def test_load_teams_from_relative_ref(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "teams.toml").write_text(
        """
[[teams]]
label = "A+B"
member_ids = ["A", "B"]

[[teams]]
label = "0+2"
member_ids = [0, 2]
""".strip(),
        encoding="utf-8",
    )

    port = TomlTeamListFilePort(data_dir=data_dir)
    teams = port.load_teams("teams")

    assert len(teams) == 2
    assert teams[0].member_ids == ("A", "B")
    assert teams[1].member_ids == (0, 2)


def test_load_ref_file_not_found_raises(tmp_path: Path) -> None:
    port = TomlCharacterListFilePort(data_dir=tmp_path / "data")
    with pytest.raises(FileNotFoundError, match="参照ファイルが見つかりません"):
        port.load_characters("characters/missing")
