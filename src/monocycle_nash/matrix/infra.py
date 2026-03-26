from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any

import numpy as np

from monocycle_nash.character import Character, MatchupVector
from monocycle_nash.loader.toml_tree import TomlTreeLoader
from monocycle_nash.matrix.base import PayoffMatrix
from monocycle_nash.team.domain import Team
from monocycle_nash.team.matrix_approx import ExactTeamPayoffCalculator

from .builder import PayoffMatrixBuilder


class MatrixFileInfrastructure:
    """行列入力のファイル読み込みと初期化を担当するインフラ層。"""

    def __init__(
        self,
        base_dir: Path | str = "data",
        entry_file: str = "data.toml",
        tree_loader: TomlTreeLoader | None = None,
    ) -> None:
        self._base_dir = Path(base_dir)
        self._entry_file = entry_file
        self._tree_loader = tree_loader or TomlTreeLoader()

    def load_matrix_input(self, matrix_name: str) -> dict[str, Any]:
        matrix_path = self._base_dir / "matrix" / matrix_name / self._entry_file
        matrix_data = self._tree_loader.load(matrix_path)
        validate_matrix_input(matrix_data)
        return matrix_data

    def load_matrix(self, matrix_name: str) -> PayoffMatrix:
        matrix_data = self.load_matrix_input(matrix_name)
        return build_matrix_from_input(matrix_data)


def build_matrix_from_input(matrix_data: dict[str, Any]) -> PayoffMatrix:
    validate_matrix_input(matrix_data)

    team_mode = _normalize_team_mode(matrix_data)
    if team_mode is not None:
        character_matrix = _build_character_matrix(matrix_data)
        teams = _build_teams(matrix_data, character_matrix)
        return _build_team_payoff_matrix(character_matrix, teams, team_mode)

    return _build_character_matrix(matrix_data)


def _build_character_matrix(matrix_data: dict[str, Any]) -> PayoffMatrix:
    labels = matrix_data.get("labels")
    if has_matrix_input(matrix_data):
        matrix = np.asarray(matrix_data["matrix"], dtype=float)
        return PayoffMatrixBuilder.from_general_matrix(matrix, labels=labels)

    return PayoffMatrixBuilder.from_characters(build_characters(matrix_data), labels=labels)


def _build_team_payoff_matrix(character_matrix: PayoffMatrix, teams: list[Team], team_mode: str) -> PayoffMatrix:
    if team_mode == "strict":
        return _build_team_payoff_matrix_strict(character_matrix, teams)

    use_monocycle_formula = team_mode == "monocycle"
    return PayoffMatrixBuilder.from_team_matchups(
        teams=teams,
        character_matrix=character_matrix,
        use_monocycle_formula=use_monocycle_formula,
    )


def _build_team_payoff_matrix_strict(character_matrix: PayoffMatrix, teams: list[Team]) -> PayoffMatrix:
    n = len(teams)
    matrix = np.zeros((n, n), dtype=float)
    calculator = ExactTeamPayoffCalculator()
    for i in range(n):
        for j in range(i + 1, n):
            value = calculator.calculate(teams[i], teams[j], character_matrix)
            matrix[i, j] = value
            matrix[j, i] = -value
    return PayoffMatrixBuilder.from_teams(matrix, teams)


def _build_teams(matrix_data: dict[str, Any], character_matrix: PayoffMatrix) -> list[Team]:
    teams_raw = matrix_data.get("teams")
    if teams_raw is None:
        return _build_default_pair_teams(character_matrix)

    teams: list[Team] = []
    for idx, team_raw in enumerate(teams_raw):
        if not isinstance(team_raw, dict):
            raise ValueError(f"teams[{idx}] はテーブルで指定してください")
        label = team_raw.get("label")
        members = team_raw.get("members")
        if not isinstance(label, str) or not label:
            raise ValueError(f"teams[{idx}].label は必須文字列です")
        if not isinstance(members, list) or not members:
            raise ValueError(f"teams[{idx}].members は1件以上の配列で指定してください")
        member_ids: list[str | int] = []
        for member in members:
            if isinstance(member, int) or (isinstance(member, str) and member):
                member_ids.append(member)
                continue
            raise ValueError(f"teams[{idx}].members は整数または空でない文字列で指定してください")
        teams.append(Team(label=label, member_ids=tuple(member_ids)))
    return teams


def _build_default_pair_teams(character_matrix: PayoffMatrix) -> list[Team]:
    strategies = character_matrix.row_strategies
    if len(strategies) < 2:
        raise ValueError("team モードでは2件以上の戦略が必要です")
    teams: list[Team] = []
    for i, j in itertools.combinations(range(len(strategies)), 2):
        left = strategies.get_strategy(i)
        right = strategies.get_strategy(j)
        teams.append(Team(label=f"{left.label}+{right.label}", member_ids=(left.id, right.id)))
    return teams


def _normalize_team_mode(matrix_data: dict[str, Any]) -> str | None:
    mode = matrix_data.get("team")
    if mode is None:
        return None
    if not isinstance(mode, str):
        raise ValueError("team は文字列で指定してください")
    if mode == "":
        return None
    return mode


def has_matrix_input(matrix_data: dict[str, Any]) -> bool:
    return "matrix" in matrix_data


def build_characters(matrix_data: dict[str, Any]) -> list[Character]:
    validate_matrix_input(matrix_data)
    if "characters" not in matrix_data:
        raise ValueError("characters 入力が必要です")

    chars_raw = matrix_data["characters"]
    characters: list[Character] = []
    for item in chars_raw:
        characters.append(
            Character(
                float(item["p"]),
                MatchupVector(float(item["v"][0]), float(item["v"][1])),
                label=item["label"],
            )
        )
    return characters


def validate_matrix_input(matrix_data: dict[str, Any]) -> None:
    has_matrix = "matrix" in matrix_data
    has_characters = "characters" in matrix_data
    if has_matrix == has_characters:
        raise ValueError("matrix または characters のどちらか片方のみ指定してください")

    labels = matrix_data.get("labels")
    if labels is not None and (not isinstance(labels, list) or not all(isinstance(x, str) for x in labels)):
        raise ValueError("labels は文字列配列で指定してください")

    if has_matrix:
        matrix = np.asarray(matrix_data["matrix"], dtype=float)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("matrix は正方2次元配列である必要があります")
        if labels is not None and len(labels) != matrix.shape[0]:
            raise ValueError("labels 数と matrix サイズが一致しません")
    else:
        chars_raw = matrix_data["characters"]
        if not isinstance(chars_raw, list) or not chars_raw:
            raise ValueError("characters は1件以上の配列が必要です")

        characters: list[Character] = []
        seen_labels: set[str] = set()
        for idx, item in enumerate(chars_raw):
            if not isinstance(item, dict):
                raise ValueError(f"characters[{idx}] はテーブルで指定してください")
            label = item.get("label")
            power = item.get("p")
            vector = item.get("v")
            if not isinstance(label, str) or not label:
                raise ValueError(f"characters[{idx}].label は必須文字列です")
            if label in seen_labels:
                raise ValueError(f"characters.label が重複しています: {label}")
            if not isinstance(power, (int, float)):
                raise ValueError(f"characters[{idx}].p は数値で指定してください")
            if not isinstance(vector, list) or len(vector) != 2:
                raise ValueError(f"characters[{idx}].v は長さ2の配列で指定してください")
            if any(not isinstance(value, (int, float)) for value in vector):
                raise ValueError(f"characters[{idx}].v は数値配列で指定してください")
            characters.append(Character(float(power), MatchupVector(float(vector[0]), float(vector[1])), label=label))
            seen_labels.add(label)

    team_mode = _normalize_team_mode(matrix_data)
    if team_mode not in (None, "strict", "2by2", "monocycle"):
        raise ValueError('team は "", "strict", "2by2", "monocycle" のいずれかで指定してください')

    teams_raw = matrix_data.get("teams")
    if teams_raw is None:
        return
    if team_mode is None:
        raise ValueError("teams を指定する場合は team モードを指定してください")
    if not isinstance(teams_raw, list) or not teams_raw:
        raise ValueError("teams は1件以上の配列で指定してください")
    seen_team_labels: set[str] = set()
    for idx, team_raw in enumerate(teams_raw):
        if not isinstance(team_raw, dict):
            raise ValueError(f"teams[{idx}] はテーブルで指定してください")
        label = team_raw.get("label")
        members = team_raw.get("members")
        if not isinstance(label, str) or not label:
            raise ValueError(f"teams[{idx}].label は必須文字列です")
        if label in seen_team_labels:
            raise ValueError(f"teams.label が重複しています: {label}")
        if not isinstance(members, list) or not members:
            raise ValueError(f"teams[{idx}].members は1件以上の配列で指定してください")
        for member in members:
            if isinstance(member, int):
                continue
            if isinstance(member, str) and member:
                continue
            raise ValueError(f"teams[{idx}].members は整数または空でない文字列で指定してください")
        seen_team_labels.add(label)
