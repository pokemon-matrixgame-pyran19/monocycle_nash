from __future__ import annotations

from dataclasses import dataclass

from ..strategy.domain import PureStrategySet


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
