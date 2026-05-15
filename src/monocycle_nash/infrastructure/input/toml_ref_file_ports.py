"""refs 経由で参照される TOML ファイル読み込みポート実装。"""

from __future__ import annotations

import csv
import tomllib
from pathlib import Path
from typing import Any

from monocycle_nash.application.ports import CharacterListFilePort, TeamListFilePort
from monocycle_nash.domain.character import Character, MatchupVector
from monocycle_nash.domain.team import Team


class _BaseTomlRefFilePort:
    def __init__(self, data_dir: Path | str = "data") -> None:
        self._data_dir = Path(data_dir)

    def _resolve_path(self, ref_path: str) -> Path:
        candidate = Path(ref_path)
        if not candidate.is_absolute():
            candidate = self._data_dir / candidate

        if candidate.is_file():
            return candidate

        if not candidate.suffix:
            with_suffix = candidate.with_suffix(".toml")
            if with_suffix.is_file():
                return with_suffix

        data_toml = candidate / "data.toml"
        if data_toml.is_file():
            return data_toml

        if not candidate.suffix:
            return candidate.with_suffix(".toml")
        return candidate

    @staticmethod
    def _load_toml(path: Path) -> dict[str, Any]:
        if not path.exists():
            raise FileNotFoundError(f"参照ファイルが見つかりません: {path}")
        with path.open("rb") as f:
            return tomllib.load(f)

    @staticmethod
    def _extract_list(data: dict[str, Any], key: str, source_path: Path) -> list[dict[str, Any]]:
        if key in data:
            raw = data[key]
        else:
            raw = data.get("params", {}).get(key)
        if not isinstance(raw, list):
            raise ValueError(f"参照ファイルの '{key}' は配列である必要があります: {source_path}")
        if not all(isinstance(item, dict) for item in raw):
            raise ValueError(
                f"参照ファイルの '{key}' 要素はテーブルである必要があります: {source_path}"
            )
        return raw


class TomlCharacterListFilePort(_BaseTomlRefFilePort, CharacterListFilePort):
    """refs.characters 向けの CharacterListFilePort 実装。"""

    def load_characters(self, path: str) -> tuple[Character, ...]:
        source_path = self._resolve_path(path)
        if source_path.suffix.lower() == ".csv":
            return self._load_characters_from_csv(source_path)
        data = self._load_toml(source_path)
        items = self._extract_list(data, "characters", source_path)

        characters: list[Character] = []
        for i, item in enumerate(items):
            try:
                power = float(item["power"])
                vector = item["vector"]
                if not isinstance(vector, list) or len(vector) != 2:
                    raise ValueError("vector は長さ2の配列である必要があります")
                label = str(item.get("label", ""))
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"characters[{i}] の形式が不正です ({exc}): {source_path}"
                ) from exc

            characters.append(
                Character(
                    power=power,
                    vector=MatchupVector(float(vector[0]), float(vector[1])),
                    label=label,
                )
            )
        return tuple(characters)

    def _load_characters_from_csv(self, source_path: Path) -> tuple[Character, ...]:
        if not source_path.exists():
            raise FileNotFoundError(f"参照ファイルが見つかりません: {source_path}")
        with source_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []
            required = ("power", "vector_x", "vector_y")
            missing = [col for col in required if col not in fieldnames]
            if missing:
                raise ValueError(
                    f"CSV 参照ファイルに必須列がありません ({', '.join(missing)}): {source_path}"
                )

            characters: list[Character] = []
            for i, row in enumerate(reader):
                if all(v is None or str(v).strip() == "" for v in row.values()):
                    continue
                try:
                    power = float(row["power"])
                    vector_x = float(row["vector_x"])
                    vector_y = float(row["vector_y"])
                    label = str(row.get("label", "") or "")
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"characters_csv[{i}] の形式が不正です ({exc}): {source_path}"
                    ) from exc

                characters.append(
                    Character(
                        power=power,
                        vector=MatchupVector(vector_x, vector_y),
                        label=label,
                    )
                )
        return tuple(characters)


class TomlTeamListFilePort(_BaseTomlRefFilePort, TeamListFilePort):
    """refs.teams 向けの TeamListFilePort 実装。"""

    def load_teams(self, path: str) -> tuple[Team, ...]:
        source_path = self._resolve_path(path)
        data = self._load_toml(source_path)
        items = self._extract_list(data, "teams", source_path)

        teams: list[Team] = []
        for i, item in enumerate(items):
            try:
                label = str(item["label"])
                member_ids = item["member_ids"]
                if not isinstance(member_ids, list):
                    raise ValueError("member_ids は配列である必要があります")
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"teams[{i}] の形式が不正です ({exc}): {source_path}"
                ) from exc

            teams.append(Team(label=label, member_ids=tuple(member_ids)))
        return tuple(teams)
