# 実装確認テスト

実装上必要な振る舞いを確認するテスト。
数学的な理論値ではなく、実装の正確性・一貫性を確認。

## テスト設計の方針

### 実装テストの性質

実装確認テストは、**実装の進行に伴って継続的に更新・追加される**ことを前提としています。

- 新しいクラスやメソッドの追加時 → 対応するテストを追加
- 実装の変更時 → テスト内容の微調整
- リファクタリング時 → テストの構造見直し

### テスト項目追加の方法

1. **新規クラス追加時**
   - 本ドキュメントの該当カテゴリにテスト項目を追加
   - 対応するテストファイルを作成
   - 既存テストとの重複を避ける

2. **既存クラスのメソッド追加時**
   - 該当カテゴリのテスト表に行を追加
   - 実装と同時にテストを作成（TDD推奨）

3. **実装変更時**
   - テスト対象の振る舞いが変わらない場合 → テストはそのまま
   - 振る舞いが変更される場合 → テストを更新し、変更内容をコメントで記録

---

## テストカテゴリ一覧

### 1. MatchupVector（相性ベクトル）

2次元ベクトルの基本演算と外積計算をテスト。

| テスト項目 | 内容 | 優先度 | 状態 |
|-----------|------|--------|------|
| 外積計算 | `v1 × v2 = -(v2 × v1)` の反対称性 | 高 | ✅ 実装済 |
| 同一ベクトル | `v × v = 0` | 高 | ✅ 実装済 |
| ベクトル加算 | `v1 + v2` が成分ごとの加算 | 高 | ⬜ 未設計 |
| ベクトル減算 | `v1 - v2` が成分ごとの減算 | 高 | ⬜ 未設計 |
| スカラー倍 | `v * s`, `s * v` が成分ごとの積 | 高 | ⬜ 未設計 |
| スカラー除算 | `v / s` が成分ごとの除算 | 高 | ⬜ 未設計 |
| 符号反転 | `-v` が成分の符号反転 | 高 | ⬜ 未設計 |
| 等価判定 | `MatchupVector`同士、配列、リストとの比較 | 中 | ⬜ 未設計 |
| 不正入力 | 2次元以外の配列を渡した場合のエラー | 中 | ⬜ 未設計 |

**テストファイル**: `tests/rule/test_character.py`

```python
# 未設計テストの例
def test_vector_addition():
    """ベクトル加算"""
    v1 = MatchupVector(1, 2)
    v2 = MatchupVector(3, 4)
    result = v1 + v2
    assert result == MatchupVector(4, 6)

def test_scalar_multiplication():
    """スカラー倍"""
    v = MatchupVector(1, 2)
    assert v * 2 == MatchupVector(2, 4)
    assert 2 * v == MatchupVector(2, 4)  # __rmul__
```

---

### 2. Character（キャラクター）

単相性モデルのキャラクター基本操作をテスト。

| テスト項目 | 内容 | 優先度 | 状態 |
|-----------|------|--------|------|
| パラメータ取得 | `p`, `v.x`, `v.y` の取得 | 高 | ✅ 実装済（暗黙的） |
| tolist変換 | デフォルト順 `[p, x, y]` | 中 | ⬜ 未設計 |
| tolist順序指定 | `order`パラメータでの順序変更 | 中 | ⬜ 未設計 |
| tolist無効キー | 無効な`order`キーでのエラー | 低 | ⬜ 未設計 |
| convert変換 | `p' = p + v × a`, `v' = v - a` の計算 | 高 | ⬜ 未設計 |
| convert連鎖 | 複数回のconvertが正しく累積 | 中 | ⬜ 未設計 |
| get_characters | データ配列からの一括生成 | 中 | ⬜ 未設計 |

**テストファイル**: `tests/rule/test_character.py`（新規作成推奨）

```python
# 未設計テストの例
def test_character_convert():
    """等パワー座標変換"""
    c = Character(1.0, MatchupVector(2, 0))
    a = MatchupVector(1, 0)
    converted = c.convert(a)
    
    # p' = p + v × a = 1.0 + (2*0 - 0*1) = 1.0
    assert converted.p == pytest.approx(1.0)
    # v' = v - a = (2-1, 0-0) = (1, 0)
    assert converted.v == MatchupVector(1, 0)
```

---

### 3. Pool / GainMatrix（利得行列）

利得行列の生成と等価性をテスト。

| テスト項目 | 内容 | 優先度 | 状態 |
|-----------|------|--------|------|
| 行列生成 | 正三角形での正しい利得計算 | 高 | ✅ 実装済 |
| 行列形状 | `size × size` の正方行列 | 高 | ✅ 実装済（暗黙的） |
| 等価判定 | 利得行列が一致すれば等価 | 高 | ✅ 実装済 |
| 回転等価 | 回転したキャラクターで同じ利得行列 | 高 | ✅ 実装済 |
| 原点移動 | `convert(a)` で等価性が保持される | 高 | ✅ 実装済 |
| 行列アクセス | `get_matrix()` の返値型・形状 | 中 | ⬜ 未設計 |
| キャラクター取得 | `get_characters()` の返値 | 中 | ⬜ 未設計 |
| pxyリスト取得 | `get_pxy_list()` のデフォルト・順序指定 | 中 | ⬜ 未設計 |

**テストファイル**: `tests/rule/test_pool.py`

```python
# 未設計テストの例
def test_matrix_access():
    """行列の取得と型確認"""
    characters = [...]
    pool = Pool(characters)
    matrix = pool.get_matrix()
    
    assert isinstance(matrix, np.ndarray)
    assert matrix.shape == (len(characters), len(characters))
```

---

### 4. BatchEnvironment（一括処理環境）

NumPyベースの高速処理バージョンをテスト。

| テスト項目 | 内容 | 優先度 | 状態 |
|-----------|------|--------|------|
| Poolとの等価性 | 同じキャラクターで同じ利得行列 | 高 | ✅ 実装済 |
| convert等価性 | BatchEnvironment同士のconvert比較 | 高 | ✅ 実装済 |
| NumPy配列入力 | ndarrayからの直接構築 | 中 | ⬜ 未設計 |
| リスト入力 | Characterリストからの構築 | 中 | ✅ 実装済（暗黙的） |
| 無効入力 | 無効な入力型でのエラー | 低 | ⬜ 未設計 |

**テストファイル**: `tests/rule/test_pool.py`

---

### 5. aCalculator（等パワー座標計算）

3キャラクターの等パワー座標計算をテスト。

| テスト項目 | 内容 | 優先度 | 状態 |
|-----------|------|--------|------|
| aベクトル計算 | 手計算と一致するa値 | 高 | ✅ 実装済 |
| 順序非依存 | キャラクター順序を変えても同じ結果 | 高 | ✅ 実装済 |
| 内部判定 | `is_inner` が正しく判定 | 高 | ✅ 実装済 |
| 境界判定 | `is_edge` が正しく判定 | 高 | ✅ 実装済 |
| 外部判定 | 外部のaで `is_inner=False` | 高 | ✅ 実装済 |
| ゼロ除算 | T=0 の場合のハンドリング | 中 | ⬜ 未設計 |
| 数値安定性 | 極端な値での計算精度 | 低 | ⬜ 未設計 |

**テストファイル**: `tests/isopower/test_calc_a.py`

---

### 6. OptimalTriangleFinder（最適三角形探索）

凸包からの最適三角形探索をテスト。

| テスト項目 | 内容 | 優先度 | 状態 |
|-----------|------|--------|------|
| 基本探索 | 正しい三角形が見つかる | 高 | ✅ 実装済 |
| 結果取得 | `get_result()` のフォーマット | 中 | ⬜ 未設計 |
| a取得 | `get_a()` で最初のaを取得 | 中 | ⬜ 未設計 |
| 最適3キャラ取得 | `get_optimal_3characters()` の返値型 | 中 | ⬜ 未設計 |
| 複数結果 | 複数三角形がある場合の処理 | 中 | ⬜ 未設計 |
| 結果なし | 有効な三角形がない場合 | 中 | ⬜ 未設計 |
| 凸包計算 | 凸包が正しく計算される | 中 | ⬜ 未設計 |

**テストファイル**: `tests/isopower/test_optimal_triangle.py`

```python
# 未設計テストの例
def test_get_a_single_result():
    """単一結果時のa取得"""
    finder = OptimalTriangleFinder(pool)
    finder.find()
    a = finder.get_a()
    
    assert isinstance(a, MatchupVector)

def test_no_valid_triangle():
    """有効な三角形がない場合"""
    # 内部にaを含む三角形がないケース
    characters = [...]
    pool = Pool(characters)
    finder = OptimalTriangleFinder(pool)
    finder.find()
    
    assert finder.get_result() == []
```

---

## 未実装クラスのテスト設計（クラス設計v2対応）

クラス設計v2で定義されているが、まだ実装・テストがないクラス群。

### 7. PayoffMatrixBuilder（利得行列ビルダー）

| テスト項目 | 内容 | 優先度 |
|-----------|------|--------|
| from_characters | CharacterリストからMonocyclePayoffMatrix生成 | 高 |
| from_general_matrix | 任意行列からGeneralPayoffMatrix生成 | 高 |
| from_teams | Teamリストからチーム利得行列生成 | 中 |
| ラベル設定 | 明示的ラベルとデフォルトラベル | 中 |

**予定テストファイル**: `tests/matrix/test_builder.py`

### 8. GeneralPayoffMatrix（一般利得行列）

| テスト項目 | 内容 | 優先度 |
|-----------|------|--------|
| 行列アクセス | `matrix`, `size`, `labels` プロパティ | 高 |
| 要素取得 | `get_value(i, j)` | 中 |
| ソルバー選択 | `solve_equilibrium()` でNashpySolver使用 | 高 |

**予定テストファイル**: `tests/matrix/test_general.py`

### 9. MonocyclePayoffMatrix（単相性モデル利得行列）

| テスト項目 | 内容 | 優先度 |
|-----------|------|--------|
| 行列計算 | `Aij = pi - pj + vi×vj` の計算 | 高 |
| characters取得 | 元のCharacterリスト保持 | 高 |
| ソルバー選択 | `solve_equilibrium()` でIsopowerSolver使用 | 高 |
| shift_origin | 等パワー座標での原点移動 | 高 |
| パワーベクトル取得 | `get_power_vector()` | 中 |
| 相性ベクトル取得 | `get_matchup_vectors()` | 中 |

**予定テストファイル**: `tests/matrix/test_monocycle.py`

### 10. NashpySolver（一般行列ソルバー）

| テスト項目 | 内容 | 優先度 |
|-----------|------|--------|
| can_solve | 全ての行列にTrueを返す | 高 |
| 均衡計算 | 線形計画法で均衡解を計算 | 高 |
| フォールバック | 均衡が見つからない場合の均等分布 | 高 |
| 対称ゲーム | 対称ゲームでの正しい均衡 | 中 |

**予定テストファイル**: `tests/solver/test_nashpy_solver.py`

### 11. IsopowerSolver（単相性モデルソルバー）

| テスト項目 | 内容 | 優先度 |
|-----------|------|--------|
| can_solve | MonocyclePayoffMatrixのみTrue | 高 |
| 最適三角形探索 | OptimalTriangleFinder使用 | 高 |
| 原点移動 | shift_origin後の均衡計算 | 高 |
| 均衡比率 | `i:j:k = Ajk:Aki:Aij` の計算 | 高 |
| フォールバック | 有効三角形なし時の均等分布 | 高 |
| 確率正規化 | 合計が1になる確率配列 | 高 |

**予定テストファイル**: `tests/solver/test_isopower_solver.py`

### 12. SolverSelector（ソルバー選択）

| テスト項目 | 内容 | 優先度 |
|-----------|------|--------|
| Monocycle選択 | MonocyclePayoffMatrix → IsopowerSolver | 高 |
| General選択 | GeneralPayoffMatrix → NashpySolver | 高 |
| solveメソッド | 選択→実行の一括処理 | 高 |

**予定テストファイル**: `tests/solver/test_selector.py`

### 13. MixedStrategy（混合戦略）

| テスト項目 | 内容 | 優先度 |
|-----------|------|--------|
| プロパティアクセス | `probabilities`, `strategy_ids` | 高 |
| 確率取得 | `get_probability(strategy_id)` | 高 |
| 検証 | `validate()` で確率合計が1か確認 | 高 |
| サポート取得 | `get_support()` で正の確率戦略を取得 | 高 |
| 不正入力 | 確率とID数が不一致の場合のエラー | 中 |

**予定テストファイル**: `tests/equilibrium/test_domain.py`

---

## 実装例

### 実装パターン

```python
def test_matchup_reverse():
    """外積の反対称性"""
    v1 = MatchupVector(random(), random())
    v2 = MatchupVector(random(), random())
    
    assert v1.times(v2) == -v2.times(v1)

def test_matrix_equality():
    """利得行列の等価判定"""
    matrix1 = MonocyclePayoffMatrix(characters)
    matrix2 = MonocyclePayoffMatrix(rotated_characters)
    
    assert matrix1 == matrix2  # 回転で一致

def test_shift_origin():
    """原点移動で等価性保持"""
    matrix = MonocyclePayoffMatrix(characters)
    shifted = matrix.shift_origin(a_vector)
    
    assert matrix == shifted  # 利得行列は不変
```

---

## 既存テストとの対応

| 既存テスト | 対応カテゴリ | 備考 |
|-----------|-------------|------|
| `test_character.py::test_Matchup_times` | MatchupVector / 外積計算 | 正三角形での検証 |
| `test_character.py::test_Matchup_reverse` | MatchupVector / 反対称性 | ランダム値での検証 |
| `test_pool.py::test_GainMatrix_set_gain` | Pool / 行列生成 | 正三角形での検証 |
| `test_pool.py::test_env_equal` | Pool / 等価判定 | 回転等価性 |
| `test_pool.py::test_env_convert` | Pool / 原点移動 | convertの等価性 |
| `test_pool.py::test_bulk_env` | BatchEnvironment / Pool等価性 | 両実装の比較 |
| `test_pool.py::test_bulk_convert` | BatchEnvironment / convert等価性 | 両実装の比較 |
| `test_calc_a.py::test_calc_a` | aCalculator / aベクトル計算 | 手計算値との比較 |
| `test_calc_a.py::test_a_order` | aCalculator / 順序非依存 | 入れ替えて同じ結果 |
| `test_calc_a.py::test_inner` | aCalculator / 内部判定 | 内部のケース |
| `test_calc_a.py::test_ounter` | aCalculator / 外部判定 | 外部のケース |
| `test_calc_a.py::test_edge` | aCalculator / 境界判定 | 辺上のケース |
| `test_optimal_triangle.py::test_finder` | OptimalTriangleFinder / 基本探索 | 期待結果の検証 |

---

## テスト追加時のチェックリスト

新しいテストを追加する際は以下を確認：

- [ ] テスト対象のクラス/メソッドが本ドキュメントの適切なカテゴリに記載されているか
- [ ] テストの目的が明確に記述されているか（コメント/DOCSTRING）
- [ ] 既存テストとの重複がないか
- [ ] テストデータが適切に設定されているか（境界値、エッジケース）
- [ ] アサーションが明確で意図が伝わるか
- [ ] テストが独立して実行可能か（他テストへの依存なし）

---

## 今後の実装予定

以下のクラス設計v2のコンポーネントは、実装時にテストを追加：

1. **matrix/**: GeneralPayoffMatrix, MonocyclePayoffMatrix, PayoffMatrixBuilder
2. **solver/**: NashpySolver, IsopowerSolver, SolverSelector
3. **equilibrium/**: MixedStrategyの拡張、validator
4. **strategy/**: ε戦略、最適反応
5. **team/**: Team関連の全機能

各実装時に本ドキュメントの「未実装クラスのテスト設計」セクションを参照し、対応するテスト項目を実装してください。
