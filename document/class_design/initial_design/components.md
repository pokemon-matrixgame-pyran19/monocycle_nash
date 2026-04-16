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
from ..team.matrix_approx import TwoPlayerTeamMatrixCalculator
from ..strategy.domain import PureStrategySet

if TYPE_CHECKING:
    from ..character.domain import Character
    from ..matrix.base import PayoffMatrix
    from ..team.domain import Team


class PayoffMatrixBuilder:
    """
    利得行列ビルダー
    - Character/Team を直接行列へ渡さず、PureStrategySetへ正規化してから構築
    """

    @staticmethod
    def from_characters(
        characters: list["Character"],
        labels: list[str] | None = None,
    ) -> MonocyclePayoffMatrix:
        return MonocyclePayoffMatrix(characters, labels=labels)

    @staticmethod
    def from_general_matrix(
        matrix: np.ndarray,
        labels: list[str] | None = None,
        row_strategies: PureStrategySet | None = None,
        col_strategies: PureStrategySet | None = None,
    ) -> GeneralPayoffMatrix:
        if row_strategies is not None:
            return GeneralPayoffMatrix(matrix, row_strategies, col_strategies)
        return GeneralPayoffMatrix(matrix, labels, col_strategies)

    @staticmethod
    def from_teams(team_payoff: np.ndarray, teams: list["Team"]) -> GeneralPayoffMatrix:
        rows = PureStrategySet.from_teams(teams, player_name="row")
        return GeneralPayoffMatrix(team_payoff, rows, rows)

    @staticmethod
    def from_team_matchups(
        teams: list["Team"],
        character_matrix: "PayoffMatrix",
        use_monocycle_formula: bool = True,
    ) -> GeneralPayoffMatrix:
        matrix_calculator = TwoPlayerTeamMatrixCalculator(
            character_matrix=character_matrix,
            use_monocycle_formula=use_monocycle_formula,
        )
        return matrix_calculator.generate_matrix(teams)
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
