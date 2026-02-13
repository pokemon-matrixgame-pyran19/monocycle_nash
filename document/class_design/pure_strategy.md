## 純粋戦略中心の設計（再整理）

### この章の結論

- **利得行列は PureStrategy しか知らない**: Character/Team は直接参照しない
- **PureStrategy 側で entity を取り込む**: `PureStrategy.from_character(...)` / `PureStrategy.from_team(...)`
- **利得行列初期化で戦略を生成する**: builder/constructor 内で row/col の戦略集合を確定
- **isopower も PureStrategy 操作へ寄せる**: Monocycle 固有処理は MonocyclePureStrategy に閉じる
- **Team拡張は初期化経路の差し替えだけ**: 行列アルゴリズムは再利用可能

### 責務分離

```text
Character / Team
    └─ ドメイン情報（label, p, v, team構成など）
        ↓ 変換
PureStrategy / MonocyclePureStrategy
    └─ 利得行列の行・列の意味を担保
        ↓ 参照
PayoffMatrix
    └─ 数値行列 + row/col戦略集合
```

### strategy/domain.py（再設計）

```python
from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from ..character.domain import Character, MatchupVector

if TYPE_CHECKING:
    from ..team.domain import Team


class StrategyEntity(Protocol):
    @property
    def label(self) -> str: ...


@dataclass(frozen=True)
class PureStrategy:
    id: str
    entity: StrategyEntity

    @classmethod
    def from_character(cls, index: int, character: Character) -> "MonocyclePureStrategy":
        return MonocyclePureStrategy(id=f"c{index}", entity=character)

    @classmethod
    def from_team(cls, index: int, team: Team) -> "PureStrategy":
        return cls(id=f"t{index}", entity=team)

    @property
    def label(self) -> str:
        return self.entity.label


@dataclass(frozen=True)
class MonocyclePureStrategy(PureStrategy):
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
    strategies: list[PureStrategy]
    player_name: str = ""

    @classmethod
    def from_characters(cls, characters: list[Character], player_name: str = "") -> "PureStrategySet":
        return cls([PureStrategy.from_character(i, c) for i, c in enumerate(characters)], player_name)

    @classmethod
    def from_teams(cls, teams: list[Team], player_name: str = "") -> "PureStrategySet":
        return cls([PureStrategy.from_team(i, t) for i, t in enumerate(teams)], player_name)

    def shift_origin(self, a: MatchupVector) -> "PureStrategySet":
        shifted = [MonocyclePureStrategy.cast(s).shift_origin(a) for s in self.strategies]
        return PureStrategySet(shifted, player_name=self.player_name)
```

### 利得行列の初期化フロー

```python
# 1) Character/Team はドメインとして準備
characters = [
    Character(power=1.0, vector=MatchupVector(1.0, 0.0), label="ピカチュウ"),
    Character(power=0.5, vector=MatchupVector(-0.5, 0.5), label="カメックス"),
]

# 2) 行列生成時に PureStrategySet を作る（Character側メソッドは使わない）
payoff = PayoffMatrixBuilder.from_characters(characters)

# 3) 行列操作は strategy 経由で解釈可能
s0 = payoff.row_strategies[0]
print(s0.id, s0.label)  # c0, ピカチュウ
```

### 行列と純粋戦略の関係

```text
PayoffMatrix
  - matrix: np.ndarray
  - row_strategies: PureStrategySet
  - col_strategies: PureStrategySet

A[i, j] の意味
  i -> row_strategies[i]
  j -> col_strategies[j]
```

### isopower 設計への反映

- `isopower` モジュールの入力は `Character` 直列ではなく `MonocyclePureStrategy` 列に統一する
- `shift_origin` は `MonocyclePureStrategy.shift_origin()` を使って strategy set 全体へ適用する
- これにより、`MonocyclePayoffMatrix` は「数値 + 戦略対応」のみ保持し、理論操作を分離できる

### Team 利得行列への拡張

- Team対応で変わるのは `PureStrategySet.from_teams()` の初期化経路のみ
- `GeneralPayoffMatrix` / `MixedStrategy` / `SolverSelector` は同一インターフェースで再利用可能
- つまり `Character行列` と `Team行列` を同じ「行列 x 純粋戦略」の枠で扱える

### 既存実装からの移行ポイント（ドキュメント方針）

1. `labels: list[str]` を行列コンストラクタの主入力にしない
2. `Character.to_pure_strategy()` 方向は採用せず、`PureStrategy.from_*` で生成する
3. `MixedStrategy.strategy_ids` は `row_strategies.ids` から作る
4. 単相性モデル固有アクセス（power/vector）は `MonocyclePureStrategy` に閉じる
```
