# クラス設計ドキュメント v2

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

1. **戦略の抽象化**: 利得行列は「戦略」に対応、戦略の実体はキャラクターまたはチーム
2. **利得行列を型で区別**: 2種類の利得行列を別クラスとして定義
3. **Solverは行列型に応じて自動選択**: Strategy Pattern
4. **単相性モデル専用の最適化**: MonocycleCharacter情報を活かした高速計算

---

## ディレクトリ構造

```
src/
├── __init__.py
├── main.py
├── character/                 # キャラクター関連
│   ├── __init__.py
│   ├── domain.py              # Character, MatchupVector
│   └── calculator.py          # パワー・ベクトル計算
├── matrix/                    # 利得行列関連（コア層）
│   ├── __init__.py
│   ├── base.py                # PayoffMatrix抽象基底クラス
│   ├── general.py             # 一般利得行列
│   ├── monocycle.py           # 単相性モデル利得行列
│   ├── pool.py                # キャラクタープール管理
│   └── builder.py             # 利得行列ビルダー
├── solver/                    # 均衡解ソルバー（Strategy Pattern）
│   ├── __init__.py
│   ├── base.py                # EquilibriumSolver抽象クラス
│   ├── selector.py            # 行列型に応じたソルバー選択
│   ├── nashpy_solver.py       # 一般行列用（nashpy線形最適化）
│   └── isopower_solver.py     # 単相性モデル用（等パワー座標高速解法）
├── isopower/                  # 等パワー座標関連（単相性モデル専用）
│   ├── __init__.py
│   ├── coordinate.py          # 座標変換
│   ├── a_calculator.py        # aベクトル計算
│   ├── triangle.py            # 最適三角形探索
│   └── evaluator.py           # 評価
├── equilibrium/               # 均衡解の表現
│   ├── __init__.py
│   ├── domain.py              # MixedStrategy
│   └── validator.py           # 均衡解の検証
├── strategy/                  # 戦略関連
│   ├── __init__.py
│   ├── epsilon.py             # ε戦略
│   └── best_response.py       # 最適反応
├── team/                      # チーム関連
│   ├── __init__.py
│   ├── domain.py              # Team
│   ├── factory.py             # チーム生成ファクトリ
│   └── matrix.py              # チーム利得行列
└── visualizer/                # 可視化
    ├── __init__.py
    ├── character.py
    ├── matrix.py
    ├── equilibrium.py
    ├── isopower.py
    └── strategy.py
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
from .base import PayoffMatrix
from solver.nashpy_solver import NashpySolver

class GeneralPayoffMatrix(PayoffMatrix):
    """
    一般の利得行列
    - 任意の行列要素を持つ
    - nashpyによる線形最適化で均衡解を計算
    """
    
    def __init__(self, matrix: np.ndarray, labels: list[str] | None = None):
        self._matrix = matrix
        self._size = matrix.shape[0]
        self._labels = labels or [f"s{i}" for i in range(self._size)]
        self._solver = NashpySolver()
    
    @property
    def matrix(self) -> np.ndarray:
        return self._matrix
    
    @property
    def size(self) -> int:
        return self._size
    
    @property
    def labels(self) -> list[str]:
        return self._labels
    
    def solve_equilibrium(self) -> "MixedStrategy":
        """nashpyによる線形最適化で均衡解を計算"""
        return self._solver.solve(self)
```

### matrix/monocycle.py

```python
import numpy as np
from typing import TYPE_CHECKING
from .base import PayoffMatrix
from character.domain import Character
from solver.isopower_solver import IsopowerSolver

if TYPE_CHECKING:
    from equilibrium.domain import MixedStrategy

class MonocyclePayoffMatrix(PayoffMatrix):
    """
    単相性モデルの利得行列
    - Character(p, v)から生成: Aij = pi - pj + vi×vj
    - 等パワー座標による高速解法を使用可能
    - 元のCharacter情報を保持
    """
    
    def __init__(self, characters: list[Character], labels: list[str] | None = None):
        self._characters = characters
        self._size = len(characters)
        self._labels = labels or [f"c{i}" for i in range(self._size)]
        self._matrix = self._calculate_matrix()
        self._solver = IsopowerSolver()
    
    @property
    def matrix(self) -> np.ndarray:
        return self._matrix
    
    @property
    def size(self) -> int:
        return self._size
    
    @property
    def labels(self) -> list[str]:
        return self._labels
    
    @property
    def characters(self) -> list[Character]:
        """元のCharacterリスト（等パワー座標計算に使用）"""
        return self._characters
    
    def _calculate_matrix(self) -> np.ndarray:
        """Aij = pi - pj + vi×vj を計算"""
        n = self._size
        A = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                A[i, j] = (
                    self._characters[i].p - self._characters[j].p +
                    self._characters[i].v.times(self._characters[j].v)
                )
        return A
    
    def solve_equilibrium(self) -> "MixedStrategy":
        """等パワー座標による高速解法で均衡解を計算"""
        return self._solver.solve(self)
    
    def shift_origin(self, a_vector: "MatchupVector") -> "MonocyclePayoffMatrix":
        """
        等パワー座標aで原点移動
        p' = p + v × a
        v' = v - a
        """
        new_characters = [c.convert(a_vector) for c in self._characters]
        return MonocyclePayoffMatrix(new_characters, self._labels)
    
    def get_power_vector(self) -> np.ndarray:
        """パワーベクトルを取得"""
        return np.array([c.p for c in self._characters])
    
    def get_matchup_vectors(self) -> np.ndarray:
        """相性ベクトルを取得 (N×2)"""
        return np.array([[c.v.x, c.v.y] for c in self._characters])
```

### matrix/builder.py

```python
import numpy as np
from typing import TYPE_CHECKING
from .general import GeneralPayoffMatrix
from .monocycle import MonocyclePayoffMatrix
from character.domain import Character

if TYPE_CHECKING:
    from team.domain import Team

class PayoffMatrixBuilder:
    """
    利得行列ビルダー
    - CharacterからMonocyclePayoffMatrixを生成
    - Teamからチーム利得行列を生成
    """
    
    @staticmethod
    def from_characters(characters: list[Character], labels: list[str] | None = None) -> MonocyclePayoffMatrix:
        """Characterリストから単相性モデル利得行列を生成"""
        return MonocyclePayoffMatrix(characters, labels)
    
    @staticmethod
    def from_general_matrix(matrix: np.ndarray, labels: list[str] | None = None) -> GeneralPayoffMatrix:
        """任意の行列から一般利得行列を生成"""
        return GeneralPayoffMatrix(matrix, labels)
    
    @staticmethod
    def from_teams(teams: list["Team"], character_matrix: MonocyclePayoffMatrix) -> GeneralPayoffMatrix:
        """
        チームリストからチーム利得行列を生成
        - チームは一般行列（Monocycleの構造を持たない）
        """
        n = len(teams)
        matrix = np.zeros((n, n))
        
        for i, team_i in enumerate(teams):
            for j, team_j in enumerate(teams):
                # チーム同士の利得を計算
                matrix[i, j] = team_i.calculate_payoff(team_j, character_matrix)
        
        labels = [t.name for t in teams]
        return GeneralPayoffMatrix(matrix, labels)
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
            return MixedStrategy(probs, matrix.labels)
        
        # 最初の均衡を使用（行プレイヤーの戦略）
        sigma_r, _ = equilibria[0]
        return MixedStrategy(sigma_r, matrix.labels)
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
        return MixedStrategy(probs, matrix.labels)
    
    def _calculate_shifted_equilibrium(
        self, 
        shifted: MonocyclePayoffMatrix, 
        triangle_result
    ) -> MixedStrategy:
        """原点移動後の均衡を計算"""
        # TODO: 三角形の頂点に対応する戦略確率を計算
        # 簡易実装: 三角形の頂点に均等分布
        n = shifted.size
        probs = np.zeros(n)
        
        # 三角形に含まれるキャラクターのインデックスを特定
        # ...
        
        return MixedStrategy(probs, shifted.labels)
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

## キャラクターと戦略の設計

### character/domain.py

```python
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from strategy.domain import PureStrategy

class MatchupVector:
    """相性ベクトル（2次元ベクトル）- 値オブジェクト"""
    - x: float
    - y: float
    - times(other): float  # 外積計算
    - 演算子オーバーロード（+, -, *, /, -v）

class Character(ABC):
    """キャラクターの抽象基底クラス"""
    
    @property
    @abstractmethod
    def label(self) -> str:
        """表示用ラベル"""
        pass
    
    @abstractmethod
    def to_strategy(self) -> "PureStrategy":
        """純粋戦略に変換"""
        pass

class MonocycleCharacter(Character):
    """
    単相性モデル用キャラクター
    - power: パワー値
    - vector: 相性ベクトル(vx, vy)
    - 等パワー座標計算に使用
    """
    - power: float
    - vector: MatchupVector
    - _label: str
    
    - __init__(power, vector, label)
    - convert(action_vector): MonocycleCharacter  # 等パワー座標で平行移動
    - tolist(order): list[float]
    - to_strategy(): PureStrategy

class GenericCharacter(Character):
    """
    一般・表示用キャラクター
    - パラメータを持たない軽量なキャラクター
    - グラフ出力時のラベル表示用
    """
    - _label: str
    
    - __init__(label)
    - to_strategy(): PureStrategy
```

### strategy/domain.py

```python
from typing import Protocol, TYPE_CHECKING, Union
from abc import ABC, abstractmethod

if TYPE_CHECKING:
    from character.domain import Character
    from team.domain import Team

class Strategy(Protocol):
    """戦略のプロトコル"""
    id: str
    label: str

class PureStrategy:
    """
    純粋戦略（利得行列の行/列に対応）
    - 実体はキャラクターまたはチーム
    """
    id: str
    label: str
    _entity: Character | Team
    
    - __init__(entity: Character | Team)
    - @property entity: Character | Team
    - is_character(): bool
    - is_team(): bool

class MixedStrategy:
    """混合戦略（ナッシュ均衡解）- ε戦略と区別"""
    probabilities: np.ndarray
    strategies: list[PureStrategy]
    
    - validate(): bool
    - get_probability(strategy_id): float
    - get_support(): list[PureStrategy]
```

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

```
利得行列 (PayoffMatrix)
    ├── 行: PureStrategy[0], PureStrategy[1], ...
    └── 列: PureStrategy[0], PureStrategy[1], ...

PureStrategy
    ├── CharacterStrategy → Character（MonocycleCharacter or GenericCharacter）
    └── TeamStrategy → Team（複数キャラクターのチーム）
```

---

## 使用例

```python
from matrix.builder import PayoffMatrixBuilder
from solver.selector import SolverSelector
from character.domain import Character, MatchupVector

# Character定義
characters = [
    Character(1.0, MatchupVector(1, 0)),
    Character(0.5, MatchupVector(-0.5, 0.5)),
    Character(0.5, MatchupVector(-0.5, -0.5)),
]

# 単相性モデル利得行列を生成
matrix = PayoffMatrixBuilder.from_characters(characters)

# 型に応じた自動選択で均衡解を計算
selector = SolverSelector()
equilibrium = selector.solve(matrix)
# → MonocyclePayoffMatrixなのでIsopowerSolverが使用される

# 一般の利得行列の場合
import numpy as np
A = np.array([[0, 1, -1], [-1, 0, 1], [1, -1, 0]])
general_matrix = PayoffMatrixBuilder.from_general_matrix(A)
equilibrium2 = selector.solve(general_matrix)
# → GeneralPayoffMatrixなのでNashpySolverが使用される
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
