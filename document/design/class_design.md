# クラス設計ドキュメント

## 概要

modeling.mdの内容と既存コードを基にしたクラス再設計案。

**設計方針**:
- フォルダ構造は「何を行うか」で機能別に階層化
- ドメイン層/アプリケーション層などのアーキテクチャ区分はドキュメント内で記述
- 厳密解/近似解など計算手法の違いはサブフォルダで分離
- 等パワー座標関連と可視化は他との関連が薄いためコロケーション

## ディレクトリ構造

```
src/
├── __init__.py
├── main.py                    # アプリケーションエントリーポイント
├── character/                 # キャラクター関連（パワー、相性ベクトル）
│   ├── __init__.py
│   ├── domain.py              # Character, MatchupVector（ドメイン層）
│   └── power_vector.py        # パワー・ベクトル計算ユーティリティ
├── matrix/                    # 利得行列関連
│   ├── __init__.py
│   ├── domain.py              # GainMatrix（ドメイン層）
│   ├── calculator.py          # 利得行列計算
│   ├── pool.py                # キャラクタープール管理
│   └── batch.py               # 高速バッチ計算（BatchEnvironment相当）
├── equilibrium/               # 均衡解関連
│   ├── __init__.py
│   ├── domain.py              # MixedStrategy（ドメイン層）
│   ├── strict.py              # 厳密解（線形最適化）
│   └── approximate.py         # 近似解（反復法など）
├── strategy/                  # 戦略関連
│   ├── __init__.py
│   ├── domain.py              # Strategyプロトコル（ドメイン層）
│   ├── epsilon.py             # ε戦略計算
│   └── best_response.py       # 最適反応計算
├── build/                     # 構築関連
│   ├── __init__.py
│   ├── domain.py              # Build（ドメイン層）
│   ├── factory.py             # 構築生成ファクトリ
│   ├── matrix.py              # 構築利得行列生成
│   └── ordering.py            # 構築順序管理
├── power/                     # 等パワー座標関連（コロケーション）
│   ├── __init__.py
│   ├── converter.py           # 原点移動・座標変換
│   ├── a_calculator.py        # aベクトル計算（厳密解）
│   ├── a_calculator_approx.py # aベクトル計算（近似解）
│   ├── triangle_finder.py     # 最適三角形探索
│   └── evaluator.py           # 三角形評価
└── visualizer/                # 可視化関連（コロケーション）
    ├── __init__.py
    ├── character.py           # キャラクター可視化
    ├── matrix.py              # 行列可視化
    ├── equilibrium.py         # 均衡解可視化
    ├── power.py               # 等パワー可視化
    └── strategy.py            # 戦略可視化
```

## アーキテクチャ層の区分（ドキュメント内管理）

### ドメイン層（domain層）
以下のファイルのクラスが該当:
- `character/domain.py`: Character, MatchupVector
- `matrix/domain.py`: GainMatrix
- `equilibrium/domain.py`: MixedStrategy
- `strategy/domain.py`: Strategy（Protocol）
- `build/domain.py`: Build

### アプリケーション層（application層）
`domain.py`以外のすべてのファイルが該当。厳密解/近似解で分離:
- `equilibrium/strict.py`: 線形計画法による厳密解
- `equilibrium/approximate.py`: 反復法による近似解
- `power/a_calculator.py`: 厳密なaベクトル計算
- `power/a_calculator_approx.py`: 近似aベクトル計算

### プレゼンテーション層
- `visualizer/` 配下すべて

---

## クラス詳細設計

### 1. character/domain.py（ドメイン層）

既存の `rule/character.py` を移動・整理。

```python
class MatchupVector:
    """相性ベクトル（2次元ベクトル）- 値オブジェクト"""
    - x: float
    - y: float
    - times(other): float  # 外積計算
    - 演算子オーバーロード（+, -, *, /, -v）

class Character:
    """キャラクターエンティティ"""
    - power: float           # パワー
    - vector: MatchupVector  # 相性ベクトル
    - label: str            # 可視化用ラベル（追加）
    - tolist(order): list[float]
    - convert(action_vector): Character  # 平行移動
```

### 2. character/power_vector.py（アプリケーション層）

```python
class PowerVectorCalculator:
    """パワー・ベクトル計算ユーティリティ"""
    - calc_power_difference(c1: Character, c2: Character): float
    - calc_matchup_score(c1: Character, c2: Character): float
```

### 3. matrix/domain.py（ドメイン層）

```python
class GainMatrix:
    """利得行列エンティティ"""
    - matrix: np.ndarray        # 利得行列
    - row_strategies: list[str] # 行戦略ID
    - col_strategies: list[str] # 列戦略ID
    - get_value(i, j): float
    - get_matrix(): np.ndarray
    - get_size(): tuple[int, int]
```

### 4. matrix/calculator.py（アプリケーション層）

```python
class GainMatrixCalculator:
    """利得行列計算"""
    - calculate(characters: list[Character]): GainMatrix
    - calculate_for_builds(builds: list[Build]): GainMatrix
```

### 5. matrix/pool.py（アプリケーション層）

既存の `rule/gain_matrix.py` (Pool) を移動・リファクタリング。

```python
class CharacterPool:
    """キャラクタープールと利得行列の管理"""
    - characters: list[Character]
    - matrix: GainMatrix
    - convert(action_vector): CharacterPool  # 平行移動
    - get_character(index): Character
    - get_pxy_list(): list[list[float]]
    - get_characters(): list[Character]
```

### 6. matrix/batch.py（アプリケーション層）

既存の `rule/gain_matrix.py` (BatchEnvironment) を移動。

```python
class BatchMatrixCalculator:
    """高速バッチ計算"""
    - calculate_vectorized(characters: np.ndarray): np.ndarray
    - convert_vectorized(characters: np.ndarray, action_vector: np.ndarray): np.ndarray
```

### 7. equilibrium/domain.py（ドメイン層）

```python
class MixedStrategy:
    """混合戦略（使用率）"""
    - probabilities: np.ndarray  # 各戦略の確率（合計100%）
    - strategy_ids: list[str]    # 戦略ID
    - validate(): bool          # 確率の合計が1になる検証
    - get_probability(strategy_id: str): float
    - normalize(): MixedStrategy  # 正規化
```

### 8. equilibrium/strict.py（アプリケーション層 - 厳密解）

既存の `rule/score.py` を統合。

```python
class StrictEquilibriumSolver:
    """厳密な均衡解計算（線形最適化）"""
    - solve(matrix: GainMatrix): MixedStrategy
    - solve_nash_equilibrium(matrix: GainMatrix): tuple[MixedStrategy, MixedStrategy]
    - calc_score(A, v1, v2): float
    - calc_weight(A, v): np.ndarray
```

### 9. equilibrium/approximate.py（アプリケーション層 - 近似解）

```python
class ApproximateEquilibriumSolver:
    """近似均衡解計算（反復法）"""
    - solve_fp(matrix: GainMatrix, iterations: int): MixedStrategy  # 仮想対戦
    - solve_replicator(matrix: GainMatrix, iterations: int): MixedStrategy  # 複製子動態
    - solve_regret_matching(matrix: GainMatrix, iterations: int): MixedStrategy  # 後悔一致
```

### 10. strategy/domain.py（ドメイン層）

```python
from typing import Protocol

class Strategy(Protocol):
    """戦略のプロトコル"""
    - id: str
    - label: str
    
class CharacterStrategy:
    """キャラクター戦略"""
    - character: Character
    
class BuildStrategy:
    """構築戦略"""
    - build: Build
```

### 11. strategy/epsilon.py（アプリケーション層）

```python
class EpsilonStrategyCalculator:
    """ε戦略計算"""
    - calculate(matrix: GainMatrix, mixed_strategy: MixedStrategy): EpsilonStrategyResult
    - find_best_response(matrix: GainMatrix, mixed_strategy: MixedStrategy): list[str]

class EpsilonStrategyResult:
    - epsilon: float            # ε値
    - best_responses: list[str] # 最適反応戦略ID
    - exploitability: float     # 搾取可能性
```

### 12. strategy/best_response.py（アプリケーション層）

```python
class BestResponseCalculator:
    """最適反応計算"""
    - calc_best_response(matrix: GainMatrix, opponent_strategy: MixedStrategy): MixedStrategy
    - calc_best_response_value(matrix: GainMatrix, opponent_strategy: MixedStrategy): float
```

### 13. build/domain.py（ドメイン層）

```python
class Build:
    """構築（複数キャラクターのチーム）"""
    - characters: list[Character]
    - name: str
    - id: str
    - get_characters(): list[Character]
    - get_size(): int
```

### 14. build/factory.py（アプリケーション層）

```python
class BuildFactory:
    """構築生成ファクトリ"""
    - create_from_indices(pool: CharacterPool, indices: list[int]): Build
    - create_all_combinations(pool: CharacterPool, team_size: int): list[Build]

class BuildRule:
    """構築ルール"""
    - team_size: int
    - allow_duplicates: bool
    - validate(build: Build): bool
```

### 15. build/matrix.py（アプリケーション層）

```python
class BuildMatrixGenerator:
    """構築利得行列生成"""
    - generate_from_character_matrix(
        char_matrix: GainMatrix,
        builds: list[Build]
      ): GainMatrix
```

### 16. build/ordering.py（アプリケーション層）

```python
class BuildOrderingService:
    """構築順序サービス"""
    - order_builds(builds: list[Build], ordering: OrderingStrategy): list[Build]
    - create_consistent_ordering(matrix: GainMatrix): list[int]
```

### 17. power/converter.py（アプリケーション層）

```python
class CoordinateConverter:
    """原点移動・座標変換"""
    - shift(characters: list[Character], action_vector: MatchupVector): list[Character]
    - shift_pool(pool: CharacterPool, action_vector: MatchupVector): CharacterPool
    - find_isopower_coordinate(characters: list[Character]): MatchupVector | None
```

### 18. power/a_calculator.py（アプリケーション層 - 厳密解）

既存の `isopower/calc_a.py` (aCalculator) を移動。

```python
class AVectorCalculator:
    """等パワーaベクトル計算（厳密解）"""
    - __init__(c1: Character, c2: Character, c3: Character)
    - calculate(): MatchupVector
    - is_inner: bool
    - is_edge: bool
```

### 19. power/a_calculator_approx.py（アプリケーション層 - 近似解）

```python
class AVectorApproxCalculator:
    """等パワーaベクトル計算（近似解）"""
    - __init__(characters: list[Character])
    - calculate_gradient_descent(): MatchupVector
    - calculate_least_squares(): MatchupVector
```

### 20. power/triangle_finder.py（アプリケーション層）

既存の `isopower/optimal_triangle.py` を移動。

```python
class OptimalTriangleFinder:
    """最適三角形探索"""
    - __init__(pool: CharacterPool)
    - find(): list[OptimalTriangleResult]
    - find_best(): OptimalTriangleResult | None

class OptimalTriangleResult:
    - a_vector: MatchupVector
    - characters: tuple[Character, Character, Character]
    - is_valid: bool
```

### 21. power/evaluator.py（アプリケーション層）

```python
class TriangleEvaluator:
    """三角形評価"""
    - evaluate_stability(result: OptimalTriangleResult): float
    - evaluate_balance(result: OptimalTriangleResult): float
```

### 22. visualizer/character.py（プレゼンテーション層）

```python
class CharacterVisualizer:
    """キャラクター可視化"""
    - plot_power_vector(characters: list[Character]): plt.Figure
    - plot_power_distribution(characters: list[Character]): plt.Figure
```

### 23. visualizer/matrix.py（プレゼンテーション層）

```python
class MatrixVisualizer:
    """行列可視化"""
    - plot_matchup_heatmap(matrix: GainMatrix): plt.Figure
    - plot_matrix_3d(matrix: GainMatrix): plt.Figure
```

### 24. visualizer/equilibrium.py（プレゼンテーション層）

```python
class EquilibriumVisualizer:
    """均衡解可視化"""
    - plot_equilibrium_distribution(mixed_strategy: MixedStrategy): plt.Figure
    - plot_convergence_history(history: list[float]): plt.Figure
```

### 25. visualizer/power.py（プレゼンテーション層）

```python
class PowerVisualizer:
    """等パワー可視化"""
    - plot_isopower_transformation(
        original: list[Character],
        transformed: list[Character],
        a_vector: MatchupVector
      ): plt.Figure
    - plot_triangle_formation(result: OptimalTriangleResult): plt.Figure
    - plot_character_convex_hull(characters: list[Character]): plt.Figure
```

### 26. visualizer/strategy.py（プレゼンテーション層）

```python
class StrategyVisualizer:
    """戦略可視化"""
    - plot_best_response_matrix(matrix: GainMatrix): plt.Figure
    - plot_epsilon_landscape(
        matrix: GainMatrix,
        strategies: list[MixedStrategy]
      ): plt.Figure
```

## 移行計画

### Phase 1: characterモジュール
- [ ] character/domain.py の作成（既存 rule/character.py から移動）
- [ ] character/power_vector.py の作成

### Phase 2: matrixモジュール
- [ ] matrix/domain.py の作成
- [ ] matrix/calculator.py の作成
- [ ] matrix/pool.py の作成（既存 rule/gain_matrix.py Poolから）
- [ ] matrix/batch.py の作成（既存 rule/gain_matrix.py BatchEnvironmentから）

### Phase 3: equilibriumモジュール
- [ ] equilibrium/domain.py の作成
- [ ] equilibrium/strict.py の作成（既存 rule/score.py から移動）
- [ ] equilibrium/approximate.py の作成

### Phase 4: strategyモジュール
- [ ] strategy/domain.py の作成
- [ ] strategy/epsilon.py の作成
- [ ] strategy/best_response.py の作成

### Phase 5: buildモジュール
- [ ] build/domain.py の作成
- [ ] build/factory.py の作成
- [ ] build/matrix.py の作成
- [ ] build/ordering.py の作成

### Phase 6: powerモジュール（コロケーション）
- [ ] power/converter.py の作成
- [ ] power/a_calculator.py の作成（既存 isopower/calc_a.py から）
- [ ] power/a_calculator_approx.py の作成
- [ ] power/triangle_finder.py の作成（既存 isopower/optimal_triangle.py から）
- [ ] power/evaluator.py の作成

### Phase 7: visualizerモジュール（コロケーション）
- [ ] visualizer/character.py の作成
- [ ] visualizer/matrix.py の作成
- [ ] visualizer/equilibrium.py の作成
- [ ] visualizer/power.py の作成
- [ ] visualizer/strategy.py の作成

### Phase 8: テスト更新
- [ ] 既存テストのパス更新
- [ ] 新規クラスのテスト作成

## 命名規則の統一

| 概念 | クラス名 | ファイル名 | 配置フォルダ |
|------|---------|-----------|-------------|
| 相性ベクトル | MatchupVector | domain.py | character/ |
| キャラクター | Character | domain.py | character/ |
| 利得行列 | GainMatrix | domain.py | matrix/ |
| キャラクタープール | CharacterPool | pool.py | matrix/ |
| 混合戦略 | MixedStrategy | domain.py | equilibrium/ |
| 厳密均衡解 | StrictEquilibriumSolver | strict.py | equilibrium/ |
| 近似均衡解 | ApproximateEquilibriumSolver | approximate.py | equilibrium/ |
| 戦略プロトコル | Strategy | domain.py | strategy/ |
| ε戦略 | EpsilonStrategyCalculator | epsilon.py | strategy/ |
| 構築 | Build | domain.py | build/ |
| 座標変換 | CoordinateConverter | converter.py | power/ |
| aベクトル（厳密） | AVectorCalculator | a_calculator.py | power/ |
| aベクトル（近似） | AVectorApproxCalculator | a_calculator_approx.py | power/ |
| 最適三角形探索 | OptimalTriangleFinder | triangle_finder.py | power/ |

## 依存関係図

```
character/domain.py（最下位）
    ↑
matrix/domain.py
    ↑
matrix/pool.py, matrix/batch.py
    ↑
matrix/calculator.py
    ↑
equilibrium/domain.py, strategy/domain.py, build/domain.py
    ↑
equilibrium/strict.py, equilibrium/approximate.py
strategy/epsilon.py, strategy/best_response.py
build/factory.py, build/matrix.py, build/ordering.py
    ↑
power/converter.py, power/a_calculator.py, power/triangle_finder.py
    ↑
visualizer/* （プレゼンテーション層）
```

## 層間の依存ルール（ドキュメント管理）

1. **domain層**（domain.pyファイル）:
   - 他のどの層にも依存しない
   - 純粋なデータ構造と基本的な振る舞いのみ

2. **application層**（domain.py以外のファイル）:
   - domain層に依存
   - 同じフォルダ内または他のapplication層に依存可能
   - visualizer層には依存しない

3. **visualizer層**:
   - domain層とapplication層に依存
   - 他のvisualizerモジュールに依存可能

## 注意点

1. **循環参照の回避**: domain層は他の層に依存しない
2. **厳密解/近似解の分離**: strict.pyとapproximate.pyで明確に分ける
3. **コロケーション**: power/とvisualizer/は関連機能を一箇所に集約
4. **既存機能の保持**: 既存の計算ロジックは移植時に変更せず、移動のみ行う
5. **テスト互換性**: 既存テストは段階的に新構造に移行
