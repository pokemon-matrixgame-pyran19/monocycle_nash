# 理論予測テスト

数学的な理論値と実装結果の一致を確認するテスト。

## 全体の方針

1. **テストビルダー**で各利得行列の理論値を一元管理
2. **単体テスト**で「入力→処理→出力」の検証
3. 入力と理想値の役割はテスト内容によって変わる

## テストビルダー

詳細: [test_builder_design.md](./test_builder_design.md)

### TheoryTestCaseが保持する値

```python
@dataclass
class TheoryTestCase:
    name: str
    powers: list[float]          # キャラクター設定
    vectors: list[tuple[float, float]]
    matrix: np.ndarray           # 利得行列（理論値）
    teams: list[list[int]] | None  # チーム構成
    team_matrix: np.ndarray | None  # チーム利得行列
    equilibrium: np.ndarray      # ナッシュ均衡（理論値）
    game_value: float | None     # 2×2ゲーム値
    approx_game_value: float | None  # 単相性近似ゲーム値
    isopower_a: tuple[float, float] | None  # 等パワー座標
    epsilon: float | None        # ε値
    description: str
```

## テストパターンと入力・出力

| テスト | 入力（テストビルダーから取得） | 出力（理想値との比較） |
|--------|------------------------------|----------------------|
| 単相性モデル行列計算 | `powers`, `vectors` | `matrix` |
| チーム行列計算 | `matrix`, `teams` | `team_matrix` |
| 一般行列の均衡解 | `matrix` | `equilibrium` |
| 単相性モデルの均衡解 | `monocycle_matrix` | `equilibrium` |
| 2×2ゲーム値計算 | `team_matrix` | `game_value` |
| 単相性近似ゲーム値 | `powers[4]`, `vectors[4]` | `approx_game_value` |
| 等パワー座標計算 | `powers[:3]`, `vectors[:3]` | `isopower_a` |
| ε値計算 | `equilibrium` | `epsilon` |

### 実装例

```python
def test_payoff_matrix():
    """単相性モデル行列計算の検証"""
    for case in TheoryTestBuilder.get_all_cases():
        # 入力: powers, vectors
        characters = build_characters(case.powers, case.vectors)
        calculated = MonocyclePayoffMatrix(characters).matrix
        
        # 理想値: matrix
        assert_matrix_close(calculated, case.matrix)

def test_team_payoff():
    """2×2ゲーム値計算の検証"""
    for case in TheoryTestBuilder.get_all_cases():
        if case.game_value is None:
            continue
            
        # 入力: team_matrix
        calculated = calculate_game_value(case.team_matrix)
        
        # 理想値: game_value
        assert_close(calculated, case.game_value)
```

## テストカテゴリ

### 1. 利得行列
- 単相性モデル: `Aij = pi - pj + vi × vj`
- チーム行列: チーム同士の利得計算

### 2. ナッシュ均衡解
- 一般行列（NashpySolver）: 線形最適化
- 単相性モデル（IsopowerSolver）: 等パワー座標

### 3. チーム利得の近似
- 2×2ゲーム値公式: `g = (ad-bc)/(a+d-b-c)`
- 単相性近似: `g = (ef+M)/((v1-v2)×(v3-v4))`

### 4. 等パワー座標（補助）
- 3点の等パワー座標計算

## 新しいテストケースの追加

1. `TheoryTestBuilder`に新しいメソッドを追加
2. 理論値を計算して各プロパティに設定
3. `get_all_cases()`に追加
4. 各テストパターンで自動的に検証される
