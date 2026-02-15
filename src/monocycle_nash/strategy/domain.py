"""
純粋戦略と戦略セットのドメインモデル。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterator, Protocol

from ..character.domain import Character, MatchupVector

if TYPE_CHECKING:
    from ..team.domain import Team


class StrategyEntity(Protocol):
    """PureStrategy がラップする実体の最小要件。"""

    @property
    def label(self) -> str:
        """表示ラベル。"""
        ...


@dataclass(frozen=True)
class LabelEntity:
    """
    ラベルのみを持つ軽量エンティティ。

    一般利得行列で Character/Team が未導入でも PureStrategy を構築するために使う。
    """

    label: str


@dataclass(frozen=True)
class PureStrategy:
    """利得行列の行/列に対応する純粋戦略。"""

    id: str
    entity: StrategyEntity

    @classmethod
    def from_character(
        cls,
        index: int,
        character: Character,
        strategy_id: str | None = None,
    ) -> "MonocyclePureStrategy":
        sid = strategy_id or f"c{index}"
        return MonocyclePureStrategy(id=sid, entity=character)

    @classmethod
    def from_team(
        cls,
        index: int,
        team: "Team",
        strategy_id: str | None = None,
    ) -> "PureStrategy":
        sid = strategy_id or f"t{index}"
        return cls(id=sid, entity=team)

    @classmethod
    def from_label(
        cls,
        index: int,
        label: str,
        strategy_id: str | None = None,
        id_prefix: str = "s",
    ) -> "PureStrategy":
        sid = strategy_id or f"{id_prefix}{index}"
        return cls(id=sid, entity=LabelEntity(label=label))

    @property
    def label(self) -> str:
        # Character 側にlabelが空文字で入っているケースとの互換のため id をフォールバックにする
        return self.entity.label or self.id

    @property
    def is_monocycle(self) -> bool:
        return isinstance(self.entity, Character)

    @property
    def power(self) -> float | None:
        return self.entity.p if isinstance(self.entity, Character) else None

    @property
    def vector(self) -> MatchupVector | None:
        return self.entity.v if isinstance(self.entity, Character) else None

    def __str__(self) -> str:
        return f"PureStrategy({self.label})"

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PureStrategy):
            return False
        return self.id == other.id


@dataclass(frozen=True)
class MonocyclePureStrategy(PureStrategy):
    """単相性モデル専用の純粋戦略。"""

    entity: Character

    @staticmethod
    def cast(strategy: PureStrategy) -> "MonocyclePureStrategy":
        if not isinstance(strategy, MonocyclePureStrategy):
            raise TypeError("MonocyclePureStrategy が必要です")
        return strategy

    @property
    def power(self) -> float:
        return self.entity.p

    @property
    def vector(self) -> MatchupVector:
        return self.entity.v

    def shift_origin(self, a: MatchupVector) -> "MonocyclePureStrategy":
        shifted = self.entity.convert(a)
        return MonocyclePureStrategy(id=self.id, entity=shifted)


@dataclass
class PureStrategySet:
    """利得行列に紐づく純粋戦略の集合。"""

    strategies: list[PureStrategy]
    player_name: str = ""

    def __post_init__(self) -> None:
        ids = [s.id for s in self.strategies]
        if len(ids) != len(set(ids)):
            raise ValueError("PureStrategySet内でIDが重複しています")

    @classmethod
    def from_characters(
        cls,
        characters: list[Character],
        player_name: str = "",
        ids: list[str] | None = None,
        id_prefix: str = "c",
    ) -> "PureStrategySet":
        if ids is not None and len(ids) != len(characters):
            raise ValueError("characters と ids の数が一致しません")
        strategies = [
            PureStrategy.from_character(
                i,
                character,
                strategy_id=ids[i] if ids is not None else f"{id_prefix}{i}",
            )
            for i, character in enumerate(characters)
        ]
        return cls(strategies=strategies, player_name=player_name)

    @classmethod
    def from_labels(
        cls,
        labels: list[str],
        player_name: str = "",
        id_prefix: str = "s",
    ) -> "PureStrategySet":
        strategies = [
            PureStrategy.from_label(i, label, strategy_id=label, id_prefix=id_prefix)
            for i, label in enumerate(labels)
        ]
        return cls(strategies=strategies, player_name=player_name)

    @classmethod
    def from_teams(
        cls,
        teams: list["Team"],
        player_name: str = "",
        ids: list[str] | None = None,
        id_prefix: str = "t",
    ) -> "PureStrategySet":
        if ids is not None and len(ids) != len(teams):
            raise ValueError("teams と ids の数が一致しません")
        strategies = [
            PureStrategy.from_team(
                i,
                team,
                strategy_id=ids[i] if ids is not None else f"{id_prefix}{i}",
            )
            for i, team in enumerate(teams)
        ]
        return cls(strategies=strategies, player_name=player_name)

    @property
    def size(self) -> int:
        return len(self.strategies)

    @property
    def labels(self) -> list[str]:
        return [s.label for s in self.strategies]

    @property
    def ids(self) -> list[str]:
        return [s.id for s in self.strategies]

    def get_strategy(self, index: int) -> PureStrategy:
        if not 0 <= index < len(self.strategies):
            raise IndexError(
                f"戦略インデックス {index} は範囲外です (0-{len(self.strategies)-1})"
            )
        return self.strategies[index]

    def get_by_id(self, strategy_id: str) -> PureStrategy | None:
        for strategy in self.strategies:
            if strategy.id == strategy_id:
                return strategy
        return None

    def get_by_label(self, label: str) -> PureStrategy | None:
        for strategy in self.strategies:
            if strategy.label == label:
                return strategy
        return None

    def shift_origin(self, a: MatchupVector) -> "PureStrategySet":
        shifted: list[PureStrategy] = []
        for strategy in self.strategies:
            shifted.append(MonocyclePureStrategy.cast(strategy).shift_origin(a))
        return PureStrategySet(strategies=shifted, player_name=self.player_name)

    def __len__(self) -> int:
        return len(self.strategies)

    def __iter__(self) -> Iterator[PureStrategy]:
        return iter(self.strategies)

    def __getitem__(self, index: int) -> PureStrategy:
        return self.get_strategy(index)
