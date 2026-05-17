# 現行 TOML 設定ファイル仕様

この文書は、現行実装で `TomlMatrixConfigPort` が読み込む TOML 設定の仕様をまとめたものです。
対象コード:

- `src/monocycle_nash/infrastructure/input/toml_matrix_config_port.py`
- `src/monocycle_nash/application/node_spec.py`
- `src/monocycle_nash/application/matrix_nodes.py`

## 1. 読み込み対象とパス解決

`load_node_spec(config_id)` の `config_id` は次を受け付けます。

- 絶対パス（拡張子あり/なし）
- `data_dir` からの相対パス（拡張子なしなら `.toml` を補完）

`method` キーがトップレベルに必須です。`name` 省略時は `"root"` になります。

## 2. ルートスキーマ

```toml
method = "..."                # 必須
name = "..."                  # 任意（既定: "root"）

[params]                        # 任意: method 固有パラメータ
# ...

[refs]                          # 任意: ファイル参照（値は文字列化される）
# characters = "..."
# teams = "..."

[children.<key>]                # 任意: 子ノード定義（再帰）
method = "..."

[[outputs]]                     # 任意: 出力定義（複数可）
method = "..."                # 必須
runner = "..."                # 任意: 同一 runner 名で最終集約
[outputs.params]                # 任意
# ...
```

## 3. `method` 一覧（行列ノード）

### `general_from_raw`

数値配列をそのまま利得行列として使うメソッド。外部ツールで計算済みの行列を読み込むときや、手入力で行列を直接指定したいときに使う。内部での計算は行わない。

| パラメータ | 必須 | 説明 |
|---|---|---|
| `params.matrix` | ✓ | N×N の二次元配列。`matrix[i][j]` は戦略 i が戦略 j に対してとる利得 |
| `params.labels` | — | 戦略名のリスト。省略時は 0 始まりの番号になる |

### `monocycle_from_characters`

キャラクターの「強さ（power）」と「特性ベクトル（vector）」から単相性モデルの利得行列を計算するメソッド。利得は `A[i,j] = power[i] - power[j] + cross(vector[i], vector[j])` で求める（`cross` は 2D 外積のスカラー）。じゃんけん的な循環構造を持つゲームの行列を作りたいときに向く。

キャラクター入力はインラインとファイル参照のいずれかで指定する。両方書いた場合は `refs.characters` が優先される。

| パラメータ | 必須 | 説明 |
|---|---|---|
| `params.characters` | △ | インライン定義。`refs.characters` がない場合に使う |
| `refs.characters` | △ | キャラクターリストのファイルパス（`params.characters` より優先、`.toml` または `.csv`） |
| `params.labels` | — | 行列ラベルの上書き。省略時はキャラクターの `label` 属性を使用 |

`params.characters` の各要素:

| フィールド | 必須 | 説明 |
|---|---|---|
| `power` | ✓ | スカラー値。大きいほど純粋に有利になる。全員同値でも可 |
| `vector` | ✓ | 2 次元ベクトル `[x, y]`。外積によってじゃんけん的な相関を生む |
| `label` | — | キャラクター名（省略可） |

`refs.characters` で `.csv` を参照する場合の列:

| 列名 | 必須 | 説明 |
|---|---|---|
| `power` | ✓ | スカラー値 |
| `vector_x` | ✓ | ベクトル x 成分 |
| `vector_y` | ✓ | ベクトル y 成分 |
| `label` | — | キャラクター名（省略可） |

### `general_from_teams_payoff`

計算済みのチーム間利得行列（`team_payoff`）とチーム定義を組み合わせて一般利得行列を構築するメソッド。チーム間の対戦結果がすでに数値で手元にある場合に使う。

チーム入力はインラインとファイル参照のいずれかで指定する。両方書いた場合は `refs.teams` が優先される。

| パラメータ | 必須 | 説明 |
|---|---|---|
| `params.team_payoff` | ✓ | チーム数×チーム数の二次元配列。`team_payoff[i][j]` はチーム i がチーム j に勝つときの利得 |
| `params.teams` | △ | インライン定義。`refs.teams` がない場合に使う |
| `refs.teams` | △ | チームリストのファイルパス（`params.teams` より優先） |

`params.teams` の各要素:

| フィールド | 必須 | 説明 |
|---|---|---|
| `label` | ✓ | チーム名 |
| `member_ids` | ✓ | メンバーのインデックスまたはラベルの配列 |

### `general_from_team_matchups`

キャラクター間の利得行列とチーム定義から、チーム間の利得行列を導出するメソッド。キャラクター同士の対戦結果を基にしてチーム戦の行列を自動計算したい場合に使う。

**入力は 2 系統ある**:

1. `children.characters` — キャラクター入力ノード（`character_inline` / `character_from_file`）
2. チーム定義（`params.teams` または `refs.teams`）

| パラメータ | 必須 | 説明 |
|---|---|---|
| `children.characters` | ✓ | キャラクター入力ノード。`method = "character_inline"` または `method = "character_from_file"` |
| `params.teams` | △ | インライン定義。`refs.teams` がない場合に使う |
| `refs.teams` | △ | チームリストのファイルパス（`params.teams` より優先） |
| `params.use_monocycle_formula` | — | `true` のとき単相性方式でチーム利得を計算（既定: `true`） |

### `character_inline`（`children.characters` 用）

| パラメータ | 必須 | 説明 |
|---|---|---|
| `params.characters` | ✓ | インラインのキャラクター定義（形式は `monocycle_from_characters` と同一） |

### `character_from_file`（`children.characters` 用）

| パラメータ | 必須 | 説明 |
|---|---|---|
| `refs.characters` | ✓ | キャラクターリストのファイルパス（`.toml` または `.csv`） |

### `random_skew_symmetric`

指定サイズのランダム交代行列（`A[i,j] = -A[j,i]`, 対角成分 0）を生成するメソッド。乱数による行列を使ったテストや実験に向く。

| パラメータ | 必須 | 説明 |
|---|---|---|
| `params.size` | ✓ | 生成する行列のサイズ N（N×N の行列になる） |
| `params.low` | — | 各要素の乱数下限（既定: `-1.0`） |
| `params.high` | — | 各要素の乱数上限（既定: `1.0`） |
| `params.seed` | — | 乱数シード。再現性を確保したい場合に指定（既定: `null`） |
| `params.max_attempts` | — | 有効な行列が生成されるまでの最大試行回数（既定: `10000`） |
| `params.labels` | — | 戦略名のリスト（省略時は 0 始まりの番号） |

### `approx_monocycle_to_general`

単相性行列（`MonocyclePayoffMatrix`）を一般行列（`GeneralPayoffMatrix`）型へ変換する近似ノード。数値は変わらず、型・内部表現だけが変わる。単相性行列を受け取れない下流処理に渡す前の型変換として使う。

| パラメータ | 必須 | 説明 |
|---|---|---|
| `children.source` | ✓ | 変換元の行列を生成する子ノード定義 |

### `approx_dominant_eigenpair`

交代行列から「支配固有値ペア」（絶対値最大の純虚固有値とその共役ペア）に対応するランク 2 成分だけを抽出するノード。元の行列を最も影響力の大きい 1 つの循環成分で近似する。`source` は交代行列を生成するノードに限る。

| パラメータ | 必須 | 説明 |
|---|---|---|
| `children.source` | ✓ | 変換元の交代行列を生成する子ノード定義 |
| `params.atol` | — | 交代行列かどうかを判定する数値許容誤差（既定: `1e-8`） |

### `approx_equilibrium_preserving`

交代行列 A を `A = J + R` と分解し、基準均衡 u に対する作用 `Au` を保つように `B = J + (p_i - p_j)` で近似するノード。「均衡（ナッシュ均衡）を保ちながら最もシンプルな単相性行列に置き換える」変換として使う。`source` は交代行列を生成するノードに限る。

| パラメータ | 必須 | 説明 |
|---|---|---|
| `children.source` | ✓ | 変換元の交代行列を生成する子ノード定義 |
| `params.atol` | — | 交代行列かどうかを判定する数値許容誤差（既定: `1e-8`） |

## 4. `outputs[].method` 一覧

### `outputs[].runner`

- 任意。同じ runner 名を指定した出力は、全ノード解決後に同一 runner でまとめて実行される
- runner 未指定時は後方互換の単体実行（出力ごとに独立）

### `payoff_directed_graph`

- 任意:
  - `outputs.params.filename`（既定: `"payoff_directed_graph.svg"`）
  - `outputs.params.threshold`（既定: `0.0`）
  - `outputs.params.canvas_size`（既定: `840`）

### `character_vector_graph`

- 任意:
  - `outputs.params.filename`（既定: `"character_vector_graph.svg"`）
  - `outputs.params.canvas_size`（既定: `840`）
  - `outputs.params.margin`（既定: `90`）

### `team_matchup_experiment_csv`

- 対象ノード: `general_from_team_matchups`
- 任意:
  - `outputs.params.filename`（既定: `"team_matchup_experiment.csv"`）
  - `outputs.params.focus_team`（既定: `0`）
    - `int`: チームインデックス
    - `str`: チームラベル
- 固定チーム i に対する各チーム j の比較実験データを CSV で出力する
  - `j3`,`j4` のベクトル（反時計回り順）
  - `v1-v2` と `v3-v4` のなす角（rad/deg）
  - `Bij`（i vs j のチーム利得）

### `team_feature_vector_csv`

- 対象ノード: `general_from_team_matchups`
- 任意:
  - `outputs.params.filename`（既定: `"team_feature_vectors.csv"`）
- 各チーム（2匹構築）の特徴ベクトルを CSV で出力する
  - 特徴ベクトル `V=(v1-v2)/(v1×v2)` の `x`,`y`
  - 原点からの距離
  - 角度（rad/deg）

### `team_feature_vector_directed_graph`

- 対象ノード: `general_from_team_matchups`
- 任意:
  - `outputs.params.filename`（既定: `"team_feature_vector_scatter_plot.svg"`）
  - `outputs.params.canvas_size`（既定: `840`）
- 各チームの特徴ベクトル `(x, y)` を2次元平面にそのまま散布図として SVG 出力する

## 5. 例

### 5.1 インライン定義（キャラクター + 出力）

```toml
method = "monocycle_from_characters"
name = "rps"

# キャラクター定義: power（強さ）と vector（特性ベクトル）でじゃんけん構造を作る
[[params.characters]]
power = 1.0
vector = [1.0, 0.0]
label = "Rock"

[[params.characters]]
power = 0.0
vector = [0.0, 1.0]
label = "Paper"

[[params.characters]]
power = -1.0
vector = [-1.0, 0.0]
label = "Scissors"

# 出力: キャラクターベクトルを SVG に描画
[[outputs]]
method = "character_vector_graph"

[outputs.params]
filename = "rps_characters.svg"
```

### 5.2 子ノード + ファイル参照

このメソッドは **2 系統の入力**を必要とする:

- `refs.teams` — チーム定義ファイルのパス（「誰と誰がチームを組むか」だけを持つ）
- `children.characters` — キャラクター入力ノード（ファイル参照またはインライン）

```toml
method = "general_from_team_matchups"
name = "team_matrix"

# 入力①: チーム定義ファイル（member_ids とラベルのみ）
[refs]
teams = "teams/default.toml"

# 入力②: キャラクター入力ノード
[children.characters]
method = "character_from_file"
name = "characters"

[children.characters.refs]
characters = "characters/default.toml"  # キャラクター定義ファイル

[params]
use_monocycle_formula = true  # 単相性方式でチーム利得を計算

# 出力: チーム間の利得有向グラフを SVG に描画
[[outputs]]
method = "payoff_directed_graph"

[outputs.params]
filename = "team_matrix.svg"
threshold = 0.0
```

### 5.3 team_matchup_experiment（CSVキャラクター参照）

大量キャラクターをインラインで持たせたくない場合は、`children.characters.refs.characters` に CSV を指定できる。

```toml
method = "general_from_team_matchups"
name = "team_matchup_experiment_all_pairs_refs_csv"

[params]
use_monocycle_formula = false

[refs]
teams = "team_matchup_experiment/teams_all_pairs.toml"

[children.characters]
method = "character_from_file"
name = "characters"

[children.characters.refs]
characters = "team_matchup_experiment/characters_grid.csv"

[[outputs]]
method = "team_matchup_experiment_csv"

[outputs.params]
filename = "team_matchup_experiment_all_pairs_refs_csv.csv"
focus_team = "team_i"
```

## 6. エラーになる代表ケース

- ルートに `method` がない
- `[[outputs]]` 要素に `method` がない
- 未知の `method` / `outputs[].method`
- 必須 `children` がない（例: `general_from_team_matchups` の `children.characters`）
