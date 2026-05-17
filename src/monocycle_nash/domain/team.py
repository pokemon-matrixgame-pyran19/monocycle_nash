from __future__ import annotations

from dataclasses import dataclass
import math

from monocycle_nash.domain.character import MatchupVector
from monocycle_nash.domain.strategy import PureStrategySet


@dataclass(frozen=True)
class TeamFeatureVector:
    """2匹構築から導出される特徴ベクトル。"""

    x: float
    y: float

    @classmethod
    def from_member_vectors(cls, v1: MatchupVector, v2: MatchupVector) -> "TeamFeatureVector":
        denom = float(v1.times(v2))
        if abs(denom) < 1e-12:
            raise ValueError(
                f"team feature vector denominator is zero: v1={v1!r}, v2={v2!r}"
            )
        diff = v1 - v2
        return cls(x=float(diff.x / denom), y=float(diff.y / denom))

    @property
    def distance_from_origin(self) -> float:
        return float(math.hypot(self.x, self.y))

    @property
    def angle_rad(self) -> float:
        return float(math.atan2(self.y, self.x))

    @property
    def angle_deg(self) -> float:
        return float(math.degrees(self.angle_rad))


@dataclass(frozen=True)
class Team:
    """構築(Team)の1戦略を表すドメイン。"""

    label: str
    member_ids: tuple[str | int, ...]

    def resolve_member_indices(self, strategies: PureStrategySet) -> list[int]:
        """member_ids(戦略ID or インデックス)を行列インデックスへ解決する。"""
        ids = strategies.ids
        resolved: list[int] = []

        for member in self.member_ids:
            if isinstance(member, int):
                if not 0 <= member < len(strategies):
                    raise IndexError(f"team member index out of range: {member}")
                resolved.append(member)
                continue

            try:
                resolved.append(ids.index(member))
            except ValueError as exc:
                raise ValueError(f"character strategy id not found: {member}") from exc

        if not resolved:
            raise ValueError("Team must contain at least one member")
        return resolved

    def calculate_feature_vector(self, strategies: PureStrategySet) -> TeamFeatureVector:
        """2匹構築から特徴ベクトル V=(v1-v2)/(v1×v2) を計算する。"""
        indices = self.resolve_member_indices(strategies)
        if len(indices) != 2:
            raise ValueError(f"Team feature vector requires exactly 2 members: {self.label}")
        s1 = strategies[indices[0]]
        s2 = strategies[indices[1]]
        if s1.vector is None or s2.vector is None:
            raise TypeError("Team feature vector requires character strategies with vectors")
        return TeamFeatureVector.from_member_vectors(s1.vector, s2.vector)
