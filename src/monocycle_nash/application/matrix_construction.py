"""Mini UC: 行列構築 - Build PayoffMatrix from various input formats."""

from __future__ import annotations

import itertools

import numpy as np

from monocycle_nash.domain.character import Character, MatchupVector
from monocycle_nash.domain.matrix.base import PayoffMatrix
from monocycle_nash.domain.matrix.builder import PayoffMatrixBuilder
from monocycle_nash.domain.team import Team
from monocycle_nash.domain.team_matrix import ExactTeamPayoffCalculator

from .dto import MatrixInputDTO


class MatrixConstructionUseCase:
    """行列構築ミニユースケース"""

    def build(self, input_dto: MatrixInputDTO) -> tuple[PayoffMatrix, list[PayoffMatrix]]:
        """
        入力DTOから利得行列を構築する。

        Returns:
            (main_matrix, intermediate_matrices) - メイン行列と中間行列のリスト
        """
        self._validate(input_dto)

        team_mode = self._normalize_team_mode(input_dto.team_mode)
        if team_mode is not None:
            character_matrix = self._build_character_matrix(input_dto)
            teams = self._build_teams(input_dto, character_matrix)
            main_matrix = self._build_team_payoff_matrix(
                character_matrix, teams, team_mode,
            )
            return main_matrix, [character_matrix]

        main_matrix = self._build_character_matrix(input_dto)
        return main_matrix, []

    def _validate(self, input_dto: MatrixInputDTO) -> None:
        has_matrix = input_dto.raw_matrix is not None
        has_characters = input_dto.characters is not None
        if has_matrix == has_characters:
            raise ValueError(
                "raw_matrix または characters のどちらか片方のみ指定してください"
            )

        if has_matrix:
            matrix = np.asarray(input_dto.raw_matrix, dtype=float)
            if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
                raise ValueError("raw_matrix は正方2次元配列である必要があります")
            if (
                input_dto.labels is not None
                and len(input_dto.labels) != matrix.shape[0]
            ):
                raise ValueError("labels 数と raw_matrix サイズが一致しません")
        else:
            assert input_dto.characters is not None
            if not input_dto.characters:
                raise ValueError("characters は1件以上の配列が必要です")
            self._validate_characters(input_dto.characters)

        team_mode = self._normalize_team_mode(input_dto.team_mode)
        if team_mode is not None and team_mode not in ("strict", "2by2", "monocycle"):
            raise ValueError(
                'team_mode は "strict", "2by2", "monocycle" のいずれかで指定してください'
            )

        if input_dto.teams is not None and team_mode is None:
            raise ValueError("teams を指定する場合は team_mode を指定してください")

    def _validate_characters(self, characters: list[dict]) -> None:
        seen_labels: set[str] = set()
        for idx, item in enumerate(characters):
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
            if any(not isinstance(v, (int, float)) for v in vector):
                raise ValueError(f"characters[{idx}].v は数値配列で指定してください")
            seen_labels.add(label)

    def _build_character_matrix(self, input_dto: MatrixInputDTO) -> PayoffMatrix:
        labels = input_dto.labels
        if input_dto.raw_matrix is not None:
            matrix = np.asarray(input_dto.raw_matrix, dtype=float)
            return PayoffMatrixBuilder.from_general_matrix(matrix, labels=labels)

        characters = self._build_characters(input_dto)
        return PayoffMatrixBuilder.from_characters(characters, labels=labels)

    def _build_characters(self, input_dto: MatrixInputDTO) -> list[Character]:
        assert input_dto.characters is not None
        characters: list[Character] = []
        for item in input_dto.characters:
            characters.append(
                Character(
                    float(item["p"]),
                    MatchupVector(float(item["v"][0]), float(item["v"][1])),
                    label=item["label"],
                )
            )
        return characters

    def _build_teams(
        self, input_dto: MatrixInputDTO, character_matrix: PayoffMatrix,
    ) -> list[Team]:
        if input_dto.teams is None:
            return self._build_default_pair_teams(character_matrix)

        teams: list[Team] = []
        for idx, team_raw in enumerate(input_dto.teams):
            label = team_raw.get("label")
            members = team_raw.get("members")
            if not isinstance(label, str) or not label:
                raise ValueError(f"teams[{idx}].label は必須文字列です")
            if not isinstance(members, list) or not members:
                raise ValueError(
                    f"teams[{idx}].members は1件以上の配列で指定してください"
                )
            member_ids: list[str | int] = []
            for member in members:
                if isinstance(member, int) or (isinstance(member, str) and member):
                    member_ids.append(member)
                    continue
                raise ValueError(
                    f"teams[{idx}].members は整数または空でない文字列で指定してください"
                )
            teams.append(Team(label=label, member_ids=tuple(member_ids)))
        return teams

    @staticmethod
    def _build_default_pair_teams(character_matrix: PayoffMatrix) -> list[Team]:
        strategies = character_matrix.row_strategies
        if len(strategies) < 2:
            raise ValueError("team モードでは2件以上の戦略が必要です")
        teams: list[Team] = []
        for i, j in itertools.combinations(range(len(strategies)), 2):
            left = strategies.get_strategy(i)
            right = strategies.get_strategy(j)
            teams.append(
                Team(
                    label=f"{left.label}+{right.label}",
                    member_ids=(left.id, right.id),
                )
            )
        return teams

    @staticmethod
    def _build_team_payoff_matrix(
        character_matrix: PayoffMatrix,
        teams: list[Team],
        team_mode: str,
    ) -> PayoffMatrix:
        if team_mode == "strict":
            return MatrixConstructionUseCase._build_team_payoff_matrix_strict(
                character_matrix, teams,
            )

        use_monocycle_formula = team_mode == "monocycle"
        return PayoffMatrixBuilder.from_team_matchups(
            teams=teams,
            character_matrix=character_matrix,
            use_monocycle_formula=use_monocycle_formula,
        )

    @staticmethod
    def _build_team_payoff_matrix_strict(
        character_matrix: PayoffMatrix,
        teams: list[Team],
    ) -> PayoffMatrix:
        n = len(teams)
        matrix = np.zeros((n, n), dtype=float)
        calculator = ExactTeamPayoffCalculator()
        for i in range(n):
            for j in range(i + 1, n):
                value = calculator.calculate(teams[i], teams[j], character_matrix)
                matrix[i, j] = value
                matrix[j, i] = -value
        return PayoffMatrixBuilder.from_teams(matrix, teams)

    @staticmethod
    def _normalize_team_mode(mode: str | None) -> str | None:
        if mode is None or mode == "":
            return None
        return mode
