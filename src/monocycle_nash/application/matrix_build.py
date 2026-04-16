"""Mini UC: 設定ファイルの内容に基づいて適切な利得行列を構築するディスパッチユースケース。"""

from __future__ import annotations

import itertools

import numpy as np

from monocycle_nash.domain.matrix.base import PayoffMatrix
from monocycle_nash.domain.matrix.builder import PayoffMatrixBuilder
from monocycle_nash.domain.team import Team
from monocycle_nash.domain.team_matrix import ExactTeamPayoffCalculator

from .matrix_build_from_characters import BuildMatrixFromCharactersUseCase
from .matrix_build_from_raw import BuildMatrixFromRawUseCase
from .ports import MatrixDataPort


class MatrixBuildUseCase:
    """
    ユースケース: 行列識別子からデータを読み込み、適切な構築ユースケースを選択して利得行列を生成する。

    - data に "characters" のみ → BuildMatrixFromCharactersUseCase（単相性モデル）
    - data に "characters" + "team_mode" → キャラクター行列を構築後、チーム行列を生成（再帰的）
    - data に "matrix" → BuildMatrixFromRawUseCase（直接読み込み）
    """

    def __init__(
        self,
        port: MatrixDataPort,
        raw_uc: BuildMatrixFromRawUseCase,
        chars_uc: BuildMatrixFromCharactersUseCase,
    ) -> None:
        self._port = port
        self._raw_uc = raw_uc
        self._chars_uc = chars_uc

    def execute(self, matrix_id: str) -> tuple[PayoffMatrix, list[PayoffMatrix]]:
        """
        行列識別子からデータを読み込み利得行列を返す。

        Returns:
            (main_matrix, intermediate_matrices)
              - チームモードの場合 intermediate_matrices にキャラクター行列が入る
              - それ以外は空リスト
        """
        data = self._port.load_matrix_data(matrix_id)
        return self._build_from_data(data)

    def _build_from_data(
        self, data: dict,
    ) -> tuple[PayoffMatrix, list[PayoffMatrix]]:
        has_characters = "characters" in data
        team_mode = _normalize_team_mode(data.get("team_mode"))

        if has_characters:
            # キャラクターから単相性モデル行列を構築（再帰的な中間ステップ）
            char_matrix = self._chars_uc.execute(data)
            if team_mode is not None:
                # チーム行列が必要 → キャラクター行列を入力として再帰的に行列構築
                teams = _build_teams(data.get("teams"), char_matrix)
                team_matrix = _build_team_payoff_matrix(char_matrix, teams, team_mode)
                return team_matrix, [char_matrix]
            return char_matrix, []

        # 生行列から直接構築
        return self._raw_uc.execute(data), []


# ---------------------------------------------------------------------------
# Helpers (team matrix construction - domain logic)
# ---------------------------------------------------------------------------


def _normalize_team_mode(mode: str | None) -> str | None:
    if mode is None or mode == "":
        return None
    if mode not in ("strict", "2by2", "monocycle"):
        raise ValueError(
            'team_mode は "strict", "2by2", "monocycle" のいずれかで指定してください',
        )
    return mode


def _build_teams(teams_raw: list[dict] | None, character_matrix: PayoffMatrix) -> list[Team]:
    if teams_raw is None:
        return _build_default_pair_teams(character_matrix)

    teams: list[Team] = []
    for idx, team_raw in enumerate(teams_raw):
        label = team_raw.get("label")
        members = team_raw.get("members")
        if not isinstance(label, str) or not label:
            raise ValueError(f"teams[{idx}].label は必須文字列です")
        if not isinstance(members, list) or not members:
            raise ValueError(
                f"teams[{idx}].members は 1 件以上の配列で指定してください",
            )
        member_ids: list[str | int] = []
        for member in members:
            if isinstance(member, int) or (isinstance(member, str) and member):
                member_ids.append(member)
                continue
            raise ValueError(
                f"teams[{idx}].members は整数または空でない文字列で指定してください",
            )
        teams.append(Team(label=label, member_ids=tuple(member_ids)))
    return teams


def _build_default_pair_teams(character_matrix: PayoffMatrix) -> list[Team]:
    strategies = character_matrix.row_strategies
    if len(strategies) < 2:
        raise ValueError("team モードでは 2 件以上の戦略が必要です")
    teams: list[Team] = []
    for i, j in itertools.combinations(range(len(strategies)), 2):
        left = strategies.get_strategy(i)
        right = strategies.get_strategy(j)
        teams.append(
            Team(
                label=f"{left.label}+{right.label}",
                member_ids=(left.id, right.id),
            ),
        )
    return teams


def _build_team_payoff_matrix(
    character_matrix: PayoffMatrix,
    teams: list[Team],
    team_mode: str,
) -> PayoffMatrix:
    if team_mode == "strict":
        return _build_team_payoff_matrix_strict(character_matrix, teams)

    use_monocycle_formula = team_mode == "monocycle"
    return PayoffMatrixBuilder.from_team_matchups(
        teams=teams,
        character_matrix=character_matrix,
        use_monocycle_formula=use_monocycle_formula,
    )


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
