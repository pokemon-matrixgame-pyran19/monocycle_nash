# 理論予測テストの設計

## 概要

特定の利得行列について、理論値をまとめて管理し、テストでは期待値と計算結果をassertする形式。

## テストビルダーの設計

### 目的

各利得行列（例：じゃんけん）について、以下を1つのオブジェクト/データとしてまとめる：

- 利得行列（期待値）
- 均衡解（混合戦略の理論値）
- キャラクター設定（パワーと相性ベクトル）
- 等パワー座標（該当する場合）

### データ構造

```python
@dataclass
class TheoryTestCase:
    """理論予測テストケース - この利得行列の理論的な正しい値を保持"""
    name: str                    # 例: "janken"
    
    # キャラクター設定（単相性モデル用）
    powers: list[float]          # パワー値 [p1, p2, ...]
    vectors: list[tuple[float, float]]  # 相性ベクトル [(vx1, vy1), ...]
    
    # 利得行列（理論値）
    matrix: np.ndarray           # 単相性モデルの利得行列
    
    # チーム設定（チームテスト用）
    teams: list[list[int]] | None  # チーム構成 [[0,1], [2,3]] など
    team_matrix: np.ndarray | None  # チーム利得行列
    
    # ナッシュ均衡（理論値）
    equilibrium: np.ndarray      # 混合戦略の確率分布
    
    # ゲーム値（チーム利得用）
    game_value: float | None     # 2×2ゲーム値公式による値
    approx_game_value: float | None  # 単相性近似による値
    
    # 等パワー座標（理論値）
    isopower_a: tuple[float, float] | None  # 等パワー座標a
    
    # ε値（理論値）
    epsilon: float | None        # 均衡解に対応するε値
    
    description: str             # 説明
```

**各プロパティの用途:**

| プロパティ | 用途 |
|-----------|------|
| `powers`, `vectors` | 単相性モデル行列計算の入力 |
| `matrix` | 単相性モデル行列計算の出力 / 一般均衡解の入力 |
| `teams` | チーム行列計算の入力 |
| `team_matrix` | チーム行列計算の出力 / 2×2ゲーム値の入力 |
| `equilibrium` | 均衡解計算の出力 / ε値計算の入力 |
| `game_value` | 2×2ゲーム値公式の出力 |
| `approx_game_value` | 単相性近似の出力 |
| `isopower_a` | 等パワー座標計算の出力 |
| `epsilon` | ε値計算の出力 |

**使い方:**
- テストビルダーは「理論的な正しい値」を保持する
- 単体テストで「これを入力にして、あれが出力になるはず」を決める
- 入力・出力の役割はテスト内容によって変わる

### テストビルダーの例

```python
class TheoryTestBuilder:
    """理論予測テストケースのビルダー"""
    
    @staticmethod
    def janken() -> TheoryTestCase:
        """じゃんけんのテストケース"""
        ROOT3 = 1.7320508075688772
        
        return TheoryTestCase(
            name="janken",
            powers=[0.0, 0.0, 0.0],
            vectors=[(2, 0), (-1, ROOT3), (-1, -ROOT3)],
            matrix=np.array([
                [0, 2*ROOT3, -2*ROOT3],
                [-2*ROOT3, 0, 2*ROOT3],
                [2*ROOT3, -2*ROOT3, 0]
            ]),
            equilibrium=np.array([1/3, 1/3, 1/3]),
            isopower_a=(0, 0),
            epsilon=None,  # TODO: 計算して設定
            description="正三角形配置の等パワー3点。均衡解は1/3ずつ。"
        )
    
    @staticmethod
    def janken_with_power_shift() -> TheoryTestCase:
        """パワー差ありじゃんけん"""
        # 理論値を設定...
        pass
    
    @staticmethod
    def get_all_cases() -> list[TheoryTestCase]:
        """全テストケースを取得"""
        return [
            TheoryTestBuilder.janken(),
            # 新しいケースをここに追加
        ]
```

## テストパターン

### パターン1: 利得行列計算の検証

入力: `powers`, `vectors` → 出力: `matrix`

```python
def test_payoff_matrix_calculation():
    """powers, vectorsからmatrixが計算できるか"""
    for case in TheoryTestBuilder.get_all_cases():
        # 入力: powers, vectors
        characters = build_characters(case.powers, case.vectors)
        calculated = MonocyclePayoffMatrix(characters).matrix
        
        # 理想値: matrix
        assert_matrix_close(calculated, case.matrix, 
                           msg=f"{case.name}: 利得行列が一致しない")
```

### パターン2: 均衡解計算の検証

入力: `matrix` → 出力: `equilibrium`

```python
def test_nash_equilibrium():
    """matrixからequilibriumが計算できるか"""
    for case in TheoryTestBuilder.get_all_cases():
        # 入力: matrix（理想値を直接使用）
        game = Game(case.matrix)
        calculated = game.solve_equilibrium()
        
        # 理想値: equilibrium
        assert_array_close(calculated.probabilities, case.equilibrium,
                          msg=f"{case.name}: 均衡解が一致しない")
```

### パターン3: ε値計算の検証

入力: `equilibrium` → 出力: `epsilon`

```python
def test_epsilon_calculation():
    """equilibriumからepsilonが計算できるか"""
    for case in TheoryTestBuilder.get_all_cases():
        if case.epsilon is None:
            continue
            
        # 入力: equilibrium（理想値を直接使用）
        strategy = MixedStrategy(case.equilibrium)
        calculated = calculate_epsilon(strategy)
        
        # 理想値: epsilon
        assert_close(calculated, case.epsilon,
                    msg=f"{case.name}: ε値が一致しない")
```

### パターン4: 等パワー座標計算の検証

入力: `powers[:3]`, `vectors[:3]` → 出力: `isopower_a`

```python
def test_isopower_coordinate():
    """3点からisopower_aが計算できるか"""
    for case in TheoryTestBuilder.get_all_cases():
        if case.isopower_a is None:
            continue
            
        # 入力: powers, vectors（先頭3点）
        characters = build_characters(case.powers[:3], case.vectors[:3])
        calculated = calculate_isopower_coordinate(characters)
        
        # 理想値: isopower_a
        assert_vector_close(calculated, case.isopower_a,
                           msg=f"{case.name}: 等パワー座標が一致しない")
```

## 新しいテストケースの追加手順

1. **データ収集**: 特定の利得行列について理論値を計算
2. **ビルダー追加**: `TheoryTestBuilder`に新しいメソッドを追加
3. **自動実行**: `get_all_cases()`に追加すれば全テストで自動的に検証

### 例: 新しいケースの追加

```python
@staticmethod
def new_example() -> TheoryTestCase:
    """新しい例"""
    return TheoryTestCase(
        name="new_example",
        powers=[...],
        vectors=[...],
        matrix=np.array([...]),
        equilibrium=np.array([...]),
        isopower_a=(..., ...),
        epsilon=...,
        description="..."
    )

# get_all_casesに追加
@staticmethod
def get_all_cases() -> list[TheoryTestCase]:
    return [
        TheoryTestBuilder.janken(),
        TheoryTestBuilder.new_example(),  # ← 追加
    ]
```

## ディレクトリ構成

```
tests/
├── theory/                    # 理論予測テスト
│   ├── __init__.py
│   ├── builder.py            # TheoryTestBuilder
│   ├── test_payoff_matrix.py # 利得行列検証
│   ├── test_equilibrium.py   # 均衡解検証
│   └── test_isopower.py      # 等パワー座標検証
├── implementation/            # 実装確認テスト
└── conftest.py               # 共通フィクスチャ
```

## メリット

1. **集中管理**: 1つの利得行列について理論値を1箇所で管理
2. **追加容易**: 新しい例を追加するだけで自動的に全パターンのテストが実行
3. **明示的**: 「じゃんけんの利得行列」vs「計算結果」という比較が明確
4. **再利用**: テストデータを他のテスト（性能テストなど）でも使用可能
