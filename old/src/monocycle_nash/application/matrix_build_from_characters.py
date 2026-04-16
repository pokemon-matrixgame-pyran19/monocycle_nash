"""Mini UC: キャラクターデータから単相性モデル利得行列を構築する。"""

from __future__ import annotations

from monocycle_nash.domain.character import Character, MatchupVector
from monocycle_nash.domain.matrix.builder import PayoffMatrixBuilder
from monocycle_nash.domain.matrix.monocycle import MonocyclePayoffMatrix


class BuildMatrixFromCharactersUseCase:
    """ユースケース: インフラ層が読み込んだキャラクターデータからドメイン層の単相性モデル利得行列を初期化する。"""

    def execute(self, data: dict) -> MonocyclePayoffMatrix:
        """
        キャラクターデータから単相性モデル利得行列を構築する。

        Args:
            data: MatrixDataPort が返す dict。"characters" キーに配列必須。

        Returns:
            MonocyclePayoffMatrix
        """
        self._validate(data)
        characters = self._build_characters(data["characters"])
        labels: list[str] | None = data.get("labels")
        return PayoffMatrixBuilder.from_characters(characters, labels=labels)

    @staticmethod
    def _validate(data: dict) -> None:
        chars = data.get("characters")
        if not isinstance(chars, list) or not chars:
            raise ValueError("data の 'characters' は 1 件以上の配列が必要です")
        seen_labels: set[str] = set()
        for idx, item in enumerate(chars):
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
                raise ValueError(f"characters[{idx}].v は長さ 2 の配列で指定してください")
            if any(not isinstance(v, (int, float)) for v in vector):
                raise ValueError(f"characters[{idx}].v は数値配列で指定してください")
            seen_labels.add(label)

    @staticmethod
    def _build_characters(chars_data: list[dict]) -> list[Character]:
        return [
            Character(
                float(c["p"]),
                MatchupVector(float(c["v"][0]), float(c["v"][1])),
                label=c["label"],
            )
            for c in chars_data
        ]
