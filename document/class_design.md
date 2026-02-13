# クラス設計ドキュメント v3

## 前提と設計方針

### 利得行列の構造

利得行列の行・列は「戦略」に対応。戦略には以下を設定可能：
- **キャラクター戦略**: 1体のキャラクター
- **チーム戦略**: 複数キャラクターのチーム（Team）

### 利得行列の2種類

**一般の利得行列 (GeneralPayoffMatrix)**
- 任意の行列要素 Aij を持つ
- 解法: nashpyによる線形最適化（汎用的だが計算コスト高）

**単相性モデルの利得行列 (MonocyclePayoffMatrix)**
- MonocycleCharacter (p, v) から生成: Aij = pi - pj + vi×vj
- 解法: 等パワー座標による高速解法（構造的性質を活用）
- 元のCharacter情報を保持

### キャラクターの2種類

**MonocycleCharacter（単相性モデル用）**
- power: パワー値
- vector: 相性ベクトル(vx, vy)
- 等パワー座標計算に使用

**GenericCharacter（一般・表示用）**
- label: 表示用ラベル
- パラメータを持たない軽量なキャラクター

### 設計方針

1. **戦略の抽象化**: 利得行列は Character/Team ではなく PureStrategy を扱う
2. **利得行列を型で区別**: 2種類の利得行列を別クラスとして定義
3. **Solverは行列型に応じて自動選択**: Strategy Pattern
4. **単相性モデル専用の最適化**: MonocyclePureStrategy のパラメータを活かした高速計算

---

## ディレクトリ構造

```
src/
├── monocycle_nash/            # パッケージルート
│   ├── __init__.py
│   ├── character/             # キャラクター関連
│   │   ├── __init__.py
│   │   ├── domain.py          # Character, MatchupVector
│   │   └── calculator.py      # パワー・ベクトル計算
│   ├── matrix/                # 利得行列関連（コア層）
│   │   ├── __init__.py
│   │   ├── base.py            # PayoffMatrix抽象基底クラス
│   │   ├── general.py         # 一般利得行列
│   │   ├── monocycle.py       # 単相性モデル利得行列
│   │   ├── pool.py            # キャラクタープール管理
│   │   └── builder.py         # 利得行列ビルダー
│   ├── solver/                # 均衡解ソルバー（Strategy Pattern）
│   │   ├── __init__.py
│   │   ├── base.py            # EquilibriumSolver抽象クラス
│   │   ├── selector.py        # 行列型に応じたソルバー選択
│   │   ├── nashpy_solver.py   # 一般行列用（nashpy線形最適化）
│   │   └── isopower_solver.py # 単相性モデル用（等パワー座標高速解法）
│   ├── isopower/              # 等パワー座標関連（単相性モデル専用）
│   │   ├── __init__.py
│   │   ├── coordinate.py      # 座標変換
│   │   ├── a_calculator.py    # aベクトル計算
│   │   ├── triangle.py        # 最適三角形探索
│   │   └── evaluator.py       # 評価
│   ├── equilibrium/           # 均衡解の表現
│   │   ├── __init__.py
│   │   ├── domain.py          # MixedStrategy
│   │   └── validator.py       # 均衡解の検証
│   ├── strategy/              # 純粋戦略関連
│   │   ├── __init__.py
│   │   └── domain.py          # PureStrategy, PureStrategySet
│   ├── team/                  # チーム関連
│   │   ├── __init__.py
│   │   ├── domain.py          # Team
│   │   ├── factory.py         # チーム生成ファクトリ
│   │   └── matrix.py          # チーム利得行列
│   └── visualizer/            # 可視化
│       ├── __init__.py
│       ├── character.py
│       ├── matrix.py
│       ├── equilibrium.py
│       ├── isopower.py
│       └── strategy.py
├── main.py                    # アプリケーションエントリーポイント
└── example.py                 # 使用例
```

**インポート例:**
```python
from monocycle_nash.character.domain import Character, MatchupVector
from monocycle_nash.matrix.monocycle import MonocyclePayoffMatrix
from monocycle_nash.solver.selector import SolverSelector
```


---

## クラス詳細設計

### matrix/base.py

```python
from abc import ABC, abstractmethod
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from equilibrium.domain import MixedStrategy

class PayoffMatrix(ABC):
    """利得行列の抽象基底クラス"""
    
    @property
    @abstractmethod
    def matrix(self) -> np.ndarray:
        """利得行列のnumpy配列"""
        pass
    
    @property
    @abstractmethod
    def size(self) -> int:
        """行列のサイズ"""
        pass
    
    @abstractmethod
    def solve_equilibrium(self) -> "MixedStrategy":
        """この行列に適した方法で均衡解を計算"""
        pass
    
    def get_value(self, i: int, j: int) -> float:
        return self.matrix[i, j]
```

### matrix/general.py

```python
import numpy as np
from typing import TYPE_CHECKING

from .base import PayoffMatrix
from ..strategy.domain import PureStrategySet

if TYPE_CHECKING:
    from ..equilibrium.domain import MixedStrategy


class GeneralPayoffMatrix(PayoffMatrix):
    """
    一般の利得行列
    - 任意の行列要素を持つ
    - 行/列は labels ではなく PureStrategySet で管理
    """

    def __init__(
        self,
        matrix: np.ndarray,
        row_strategies: PureStrategySet,
        col_strategies: PureStrategySet | None = None,
    ):
        self._matrix = np.array(matrix, dtype=float)
        self._row_strategies = row_strategies
        self._col_strategies = col_strategies or row_strategies

        if self._matrix.shape != (len(self._row_strategies), len(self._col_strategies)):
            raise ValueError("行列サイズと戦略数が一致しません")

    @property
    def matrix(self) -> np.ndarray:
        return self._matrix

    @property
    def size(self) -> int:
        # 対称ゲームでは row size == col size
        return len(self._row_strategies)

    @property
    def row_strategies(self) -> PureStrategySet:
        return self._row_strategies

    @property
    def col_strategies(self) -> PureStrategySet:
        return self._col_strategies

    def solve_equilibrium(self) -> "MixedStrategy":
        from ..solver.selector import SolverSelector
        selector = SolverSelector()
        return selector.solve(self)
```

### matrix/monocycle.py

```python
import numpy as np
from typing import TYPE_CHECKING

from .base import PayoffMatrix
from ..character.domain import MatchupVector
from ..strategy.domain import MonocyclePureStrategy, PureStrategySet

if TYPE_CHECKING:
    from ..character.domain import Character
    from ..equilibrium.domain import MixedStrategy


class MonocyclePayoffMatrix(PayoffMatrix):
    """
    単相性モデルの利得行列
    - Aij = pi - pj + vi×vj
    - 行/列は MonocyclePureStrategy で保持
    - 行列操作は Matrix x PureStrategy で完結
    """

    def __init__(
        self,
        row_strategies: PureStrategySet,
        col_strategies: PureStrategySet | None = None,
    ):
        self._row_strategies = row_strategies
        self._col_strategies = col_strategies or row_strategies
        self._matrix = self._calculate_matrix()

    @classmethod
    def from_characters(cls, characters: list["Character"]) -> "MonocyclePayoffMatrix":
        rows = PureStrategySet.from_characters(characters, player_name="row")
        return cls(row_strategies=rows, col_strategies=rows)

    @property
    def matrix(self) -> np.ndarray:
        return self._matrix

    @property
    def size(self) -> int:
        return len(self._row_strategies)

    @property
    def row_strategies(self) -> PureStrategySet:
        return self._row_strategies

    @property
    def col_strategies(self) -> PureStrategySet:
        return self._col_strategies

    def _calculate_matrix(self) -> np.ndarray:
        n_row = len(self._row_strategies)
        n_col = len(self._col_strategies)
        A = np.zeros((n_row, n_col))

        for i, row in enumerate(self._row_strategies):
            for j, col in enumerate(self._col_strategies):
                r = MonocyclePureStrategy.cast(row)
                c = MonocyclePureStrategy.cast(col)
                A[i, j] = r.power - c.power + r.vector.times(c.vector)
        return A

    def shift_origin(self, a_vector: MatchupVector) -> "MonocyclePayoffMatrix":
        shifted_rows = self._row_strategies.shift_origin(a_vector)
        shifted_cols = self._col_strategies.shift_origin(a_vector)
        return MonocyclePayoffMatrix(shifted_rows, shifted_cols)

    def solve_equilibrium(self) -> "MixedStrategy":
        from ..solver.selector import SolverSelector
        selector = SolverSelector()
        return selector.solve(self)
```

### matrix/builder.py

```python
import numpy as np
from typing import TYPE_CHECKING

from .general import GeneralPayoffMatrix
from .monocycle import MonocyclePayoffMatrix
from ..strategy.domain import PureStrategySet

if TYPE_CHECKING:
    from ..character.domain import Character
    from ..team.domain import Team


class PayoffMatrixBuilder:
    """
    利得行列ビルダー
    - Character/Team を直接行列へ渡さず、PureStrategySetへ正規化してから構築
    """

    @staticmethod
    def from_characters(characters: list["Character"]) -> MonocyclePayoffMatrix:
        rows = PureStrategySet.from_characters(characters, player_name="row")
        return MonocyclePayoffMatrix(rows)

    @staticmethod
    def from_general_matrix(
        matrix: np.ndarray,
        row_strategies: PureStrategySet,
        col_strategies: PureStrategySet | None = None,
    ) -> GeneralPayoffMatrix:
        return GeneralPayoffMatrix(matrix, row_strategies, col_strategies)

    @staticmethod
    def from_teams(team_payoff: np.ndarray, teams: list["Team"]) -> GeneralPayoffMatrix:
        rows = PureStrategySet.from_teams(teams, player_name="row")
        return GeneralPayoffMatrix(team_payoff, rows, rows)
```

### solver/base.py

```python
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from matrix.base import PayoffMatrix
    from equilibrium.domain import MixedStrategy

class EquilibriumSolver(ABC):
    """均衡解ソルバーの抽象基底クラス"""
    
    @abstractmethod
    def solve(self, matrix: "PayoffMatrix") -> "MixedStrategy":
        """利得行列から均衡解を計算"""
        pass
    
    @abstractmethod
    def can_solve(self, matrix: "PayoffMatrix") -> bool:
        """このソルバーで解けるか判定"""
        pass
```

### solver/nashpy_solver.py

```python
import numpy as np
import nashpy as nash
from .base import EquilibriumSolver
from matrix.general import GeneralPayoffMatrix
from equilibrium.domain import MixedStrategy

class NashpySolver(EquilibriumSolver):
    """
    nashpyによる線形最適化ソルバー
    - 一般の利得行列に対応
    - 線形計画法で厳密解を求める
    """
    
    def can_solve(self, matrix) -> bool:
        """全ての行列に適用可能"""
        return True
    
    def solve(self, matrix: GeneralPayoffMatrix) -> MixedStrategy:
        """
        nashpyでナッシュ均衡を計算
        対称ゲームを仮定し、行プレイヤーの均衡戦略を返す
        """
        A = matrix.matrix
        game = nash.Game(A)
        
        # 線形計画法で均衡を計算
        equilibria = list(game.linear_program())
        
        if not equilibria:
            # 均衡が見つからない場合は均等分布を返す
            probs = np.ones(matrix.size) / matrix.size
            return MixedStrategy(probs, matrix.row_strategies.ids)
        
        # 最初の均衡を使用（行プレイヤーの戦略）
        sigma_r, _ = equilibria[0]
        return MixedStrategy(sigma_r, matrix.row_strategies.ids)
```

### solver/isopower_solver.py

```python
import numpy as np
from .base import EquilibriumSolver
from matrix.monocycle import MonocyclePayoffMatrix
from equilibrium.domain import MixedStrategy
from isopower.coordinate import find_isopower_coordinate
from isopower.triangle import OptimalTriangleFinder

class IsopowerSolver(EquilibriumSolver):
    """
    等パワー座標による高速ソルバー
    - 単相性モデル専用
    - 等パワー座標を使った高速解法
    """
    
    def can_solve(self, matrix) -> bool:
        """MonocyclePayoffMatrixのみ対応"""
        return isinstance(matrix, MonocyclePayoffMatrix)
    
    def solve(self, matrix: MonocyclePayoffMatrix) -> MixedStrategy:
        """
        等パワー座標による高速解法で均衡解を計算
        
        アルゴリズム:
        1. 最適な等パワー座標aを探索
        2. aで原点移動
        3. 移動後の空間で均衡を計算
        """
        # 最適三角形を探索
        finder = OptimalTriangleFinder(matrix)
        result = finder.find_best()
        
        if result is None:
            # 有効な三角形がない場合はフォールバック
            return self._fallback_solve(matrix)
        
        # 等パワー座標で原点移動
        shifted = matrix.shift_origin(result.a_vector)
        
        # 移動後の均衡を計算（単純化された構造）
        return self._calculate_shifted_equilibrium(shifted, result)
    
    def _fallback_solve(self, matrix: MonocyclePayoffMatrix) -> MixedStrategy:
        """フォールバック: 均等分布を返す"""
        probs = np.ones(matrix.size) / matrix.size
        return MixedStrategy(probs, matrix.row_strategies.ids)
    
    def _calculate_shifted_equilibrium(
        self, 
        shifted: MonocyclePayoffMatrix, 
        triangle_result
    ) -> MixedStrategy:
        """
        原点移動後の均衡を計算
        
        理論: i,j,kの組み合わせに対し、混合戦略の比率は
        i:j:k = Ajk : Aki : Aij
        
        ここで Aij は移動後の利得行列の要素
        """
        i, j, k = triangle_result.indices
        A = shifted.matrix
        
        # 比率を計算: i:j:k = A[j,k] : A[k,i] : A[i,j]
        ratio_i = A[j, k]
        ratio_j = A[k, i]
        ratio_k = A[i, j]
        
        # 確率に正規化（合計が1になるように）
        total = ratio_i + ratio_j + ratio_k
        prob_i = ratio_i / total
        prob_j = ratio_j / total
        prob_k = ratio_k / total
        
        # 全戦略の確率配列を作成（ijk以外は0）
        n = shifted.size
        probs = np.zeros(n)
        probs[i] = prob_i
        probs[j] = prob_j
        probs[k] = prob_k
        
        return MixedStrategy(probs, shifted.row_strategies.ids)
```

### solver/selector.py

```python
from .base import EquilibriumSolver
from matrix.base import PayoffMatrix
from matrix.general import GeneralPayoffMatrix
from matrix.monocycle import MonocyclePayoffMatrix
from .nashpy_solver import NashpySolver
from .isopower_solver import IsopowerSolver

class SolverSelector:
    """
    利得行列の型に応じて最適なソルバーを選択
    - Strategy Patternによる自動選択
    """
    
    def __init__(self):
        self._isopower_solver = IsopowerSolver()
        self._nashpy_solver = NashpySolver()
    
    def select(self, matrix: PayoffMatrix) -> EquilibriumSolver:
        """
        行列型に応じたソルバーを選択
        - MonocyclePayoffMatrix → IsopowerSolver（高速）
        - GeneralPayoffMatrix → NashpySolver（汎用）
        """
        if isinstance(matrix, MonocyclePayoffMatrix):
            return self._isopower_solver
        return self._nashpy_solver
    
    def solve(self, matrix: PayoffMatrix):
        """適切なソルバーで均衡解を計算"""
        solver = self.select(matrix)
        return solver.solve(matrix)
```

### isopower/a_calculator.py

```python
class IsopowerACalculator:
    """
    等パワー座標 a の計算
    
    理論: 任意の3つの戦略 i, j, k を等パワーにする座標 a は
    
    a = (p_i(v_j - v_k) + p_j(v_k - v_i) + p_k(v_i - v_j)) / T
    
    ここで T は以下の 3×3 行列式:
    T = | p_i  p_j  p_k |
        | v_i  v_j  v_k |
    
    v は2次元ベクトルなので、これで3×3の行列式となる。
    分子はベクトル、分母はスカラーなので、a はベクトルとなる。
    """
    
    - calculate_a_vector(c_i: MonocycleCharacter, 
                        c_j: MonocycleCharacter, 
                        c_k: MonocycleCharacter) -> MatchupVector
    - calculate_t_determinant(c_i, c_j, c_k) -> float  # Tの計算
    - calculate_numerator(c_i, c_j, c_k) -> MatchupVector  # 分子の計算
    - is_inner: bool  # aが三角形の内部にあるか（ナッシュ均衡判定に使用）
```

### isopower/triangle.py

```python
class OptimalTriangleFinder:
    """
    最適な3点（ナッシュ均衡を形成する三角形）を探索
    
    探索アルゴリズム:
    1. キャラクターの凸包（ConvexHull）を計算
    2. 凸包の各面（三角形）に対して:
       a. その3点で等パワー座標 a を計算
       b. a が三角形の内部にあるか判定（is_inner）
    3. 内部にaがある三角形を最適な組み合わせとして返す
    
    理論的根拠:
    - 平行移動前の点 vi, vj, vk の内側に a がある
    - または同じ意味だが平行移動後の vi, vj, vk が原点を囲む
    - この条件を満たすとき、等パワーの3点が最大パワーになりナッシュ均衡となる
    """
    
    - __init__(pool: Pool)
    - find() -> None  # 探索実行
    - get_result() -> list[list]  # 結果取得 [a, c_i, c_j, c_k] のリスト
    - get_a() -> MatchupVector  # 最初の結果のaを返す
    - get_optimal_3characters() -> set[Character]  # 最適3キャラクター
```

### equilibrium/domain.py

```python
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from matrix.base import PayoffMatrix

class MixedStrategy:
    """混合戦略（ナッシュ均衡解）"""
    
    def __init__(self, probabilities: np.ndarray, strategy_ids: list[str]):
        if len(probabilities) != len(strategy_ids):
            raise ValueError("確率とIDの数が一致しません")
        
        self._probs = probabilities
        self._ids = strategy_ids
    
    @property
    def probabilities(self) -> np.ndarray:
        return self._probs
    
    @property
    def strategy_ids(self) -> list[str]:
        return self._ids
    
    def get_probability(self, strategy_id: str) -> float:
        """特定戦略の確率を取得"""
        try:
            idx = self._ids.index(strategy_id)
            return self._probs[idx]
        except ValueError:
            return 0.0
    
    def validate(self, tolerance: float = 1e-6) -> bool:
        """確率の合計が1か検証"""
        return abs(np.sum(self._probs) - 1.0) < tolerance
    
    def get_support(self, threshold: float = 1e-6) -> list[str]:
        """サポート（正の確率を持つ戦略）を取得"""
        return [self._ids[i] for i, p in enumerate(self._probs) if p > threshold]
```

---

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

---

## 利得行列の型階層

```
PayoffMatrix (抽象基底)
    ├── GeneralPayoffMatrix      # 一般の利得行列
    │   └── 解法: NashpySolver (線形最適化)
    │
    └── MonocyclePayoffMatrix    # 単相性モデル
        ├── MonocycleCharacter情報を保持
        ├── 等パワー座標による高速解法: IsopowerSolver
        └── shift_origin() で原点移動可能
```

### 利得行列と戦略の関係

```text
利得行列 (PayoffMatrix)
    ├── 行: row_strategies[i] = PureStrategy
    └── 列: col_strategies[j] = PureStrategy

PureStrategy
    ├── MonocyclePureStrategy -> Character(label, p, v)
    └── PureStrategy -> Team(label, members)
```

---

## 使用例

```python
from monocycle_nash.matrix.builder import PayoffMatrixBuilder
from monocycle_nash.solver.selector import SolverSelector
from monocycle_nash.character.domain import Character, MatchupVector

characters = [
    Character(1.0, MatchupVector(1, 0), label="A"),
    Character(0.5, MatchupVector(-0.5, 0.5), label="B"),
    Character(0.5, MatchupVector(-0.5, -0.5), label="C"),
]

matrix = PayoffMatrixBuilder.from_characters(characters)
selector = SolverSelector()
equilibrium = selector.solve(matrix)

# strategy id は row_strategies に紐づく
print(matrix.row_strategies.ids)
```

---

## チーム利得行列の近似計算（追加設計）

### 概要

チーム利得行列の作成において、通常はキャラクター対戦のゲームを解いて値を設定するが、**二人チーム（2×2利得行列）**の場合は公式を用いて高速に計算可能。さらに**単相性モデルの仮定**を追加すると、より簡潔な計算が可能になる。

### team/matrix_approx.py

```python
class TwoPlayerTeamMatrixApproximator:
    """
    二人チーム利得行列の近似計算
    - 2×2利得行列のゲーム値公式を使用して高速化
    - 単相性モデル仮定でさらに最適化
    """
    - __init__(character_matrix: PayoffMatrix)
    - calculate_team_value(team1: Team, team2: Team): float
    - generate_approx_matrix(teams: list[Team]): PayoffMatrix

class TwoByTwoGameValueCalculator:
    """
    2×2利得行列のゲーム値計算
    - 公式: g = (ad - bc) / (a + d - b - c)
    """
    - calculate(matrix: np.ndarray) -> float
    - calculate_saddle_point(matrix: np.ndarray) -> tuple[float, float] | None

class MonocycleTwoByTwoApproximator:
    """
    単相性モデル仮定下的2×2ゲーム値計算
    - パラメータ: p1, p2, p3, p4, v1, v2, v3, v4
    - 公式: g = (ef + M) / ((v1-v2) × (v3-v4))
      ここで e = p1-p2 + v1×v2, f = p3-p4 + v3×v4
      M は4×4行列式 |1 1 1 1; p1 p2 p3 p4; v1 v2 v3 v4|
    """
    - __init__(c1: MonocycleCharacter, c2: MonocycleCharacter, 
               c3: MonocycleCharacter, c4: MonocycleCharacter)
    - calculate_game_value(): float
    - calculate_e_parameter(): float
    - calculate_f_parameter(): float
    - calculate_m_determinant(): float

class TeamPayoffFormulaSelector:
    """
    チーム利得行列生成時の計算方法選択
    - Strategy Patternで厳密解/近似解を自動選択
    """
    - __init__(use_monocycle_approx: bool = True)
    - select_calculator(team1: Team, team2: Team, 
                        char_matrix: PayoffMatrix) -> TeamPayoffCalculator
    - calculate(team1: Team, team2: Team, char_matrix: PayoffMatrix): float

class TeamPayoffCalculator(ABC):
    """チーム利得計算の抽象基底クラス"""
    - @abstractmethod calculate(team1: Team, team2: Team, 
                                char_matrix: PayoffMatrix): float

class ExactTeamPayoffCalculator(TeamPayoffCalculator):
    """厳密なチーム利得計算（全キャラクター対戦を解く）"""
    - calculate(team1: Team, team2: Team, 
                char_matrix: PayoffMatrix): float

class TwoByTwoApproximateCalculator(TeamPayoffCalculator):
    """2×2ゲーム値公式による近似計算"""
    - calculate(team1: Team, team2: Team, 
                char_matrix: PayoffMatrix): float

class MonocycleApproximateCalculator(TeamPayoffCalculator):
    """単相性モデル仮定による最速近似計算"""
    - calculate(team1: Team, team2: Team, 
                char_matrix: MonocyclePayoffMatrix): float
```

### 利得行列近似計算のクラス階層

```
TeamPayoffCalculator (抽象基底)
    ├── ExactTeamPayoffCalculator          # 厳密解（全対戦解く）
    ├── TwoByTwoApproximateCalculator      # 2×2公式による近似
    │   └── TwoByTwoGameValueCalculator    # g = (ad-bc)/(a+d-b-c)
    └── MonocycleApproximateCalculator     # 単相性モデル仮定による最速近似
        └── MonocycleTwoByTwoApproximator  # g = (ef+M)/((v1-v2)×(v3-v4))

TwoPlayerTeamMatrixApproximator
    └── TeamPayoffFormulaSelector          # 計算方法の自動選択
        ├── ExactTeamPayoffCalculator      # フォールバック用
        ├── TwoByTwoApproximateCalculator  # 一般2×2行列用
        └── MonocycleApproximateCalculator # 単相性モデル用
```

---

## メリット

1. **型安全性**: 行列の型で解法を保証
2. **自動最適化**: MonocyclePayoffMatrixは自動的に高速解法を使用
3. **拡張性**: 新しい行列型・解法を追加しやすい
4. **明確な責務分離**: 
   - `matrix/`: 行列の構造とデータ
   - `solver/`: 解法アルゴリズム
   - `isopower/`: 単相性モデル専用の最適化
5. **チーム利得行列の高速化**: 
   - 二人チームでは2×2公式でO(1)計算
   - 単相性モデル仮定でさらに効率化


