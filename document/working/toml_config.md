# 現行 TOML 設定ファイル仕様

この文書は、現行実装で `TomlMatrixConfigPort` が読み込む TOML 設定の仕様をまとめたものです。
対象コード:

- `/home/runner/work/monocycle_nash/monocycle_nash/src/monocycle_nash/infrastructure/input/toml_matrix_config_port.py`
- `/home/runner/work/monocycle_nash/monocycle_nash/src/monocycle_nash/application/node_spec.py`
- `/home/runner/work/monocycle_nash/monocycle_nash/src/monocycle_nash/application/matrix_nodes.py`

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
[outputs.params]                # 任意
# ...
```

## 3. `method` 一覧（行列ノード）

### `general_from_raw`

- 必須: `params.matrix`
- 任意: `params.labels`

### `monocycle_from_characters`

- キャラクター入力（どちらか）
  - `params.characters`（インライン）
  - `refs.characters`（ファイル参照）
- 任意: `params.labels`
- `refs.characters` と `params.characters` が両方ある場合、`refs.characters` を優先

`params.characters` の各要素:

- `power` (float)
- `vector` ([x, y])
- `label` (省略可)

### `general_from_teams_payoff`

- 必須: `params.team_payoff`
- チーム入力（どちらか）
  - `params.teams`（インライン）
  - `refs.teams`（ファイル参照）
- `refs.teams` と `params.teams` が両方ある場合、`refs.teams` を優先

`params.teams` の各要素:

- `label`
- `member_ids` (配列)

### `general_from_team_matchups`

- 必須: `children.character_matrix`（子ノード）
- チーム入力（どちらか）
  - `params.teams`
  - `refs.teams`
- 任意: `params.use_monocycle_formula`（既定: `true`）

### `random_skew_symmetric`

- 必須: `params.size`
- 任意:
  - `params.low`（既定: `-1.0`）
  - `params.high`（既定: `1.0`）
  - `params.seed`（既定: `null`）
  - `params.max_attempts`（既定: `10000`）
  - `params.labels`

### `approx_monocycle_to_general`

- 必須: `children.source`

### `approx_dominant_eigenpair`

- 必須: `children.source`
- 任意: `params.atol`（既定: `1e-8`）

### `approx_equilibrium_preserving`

- 必須: `children.source`
- 任意: `params.atol`（既定: `1e-8`）

## 4. `outputs[].method` 一覧

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

## 5. 例

### 5.1 インライン定義（キャラクター + 出力）

```toml
method = "monocycle_from_characters"
name = "rps"

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

[[outputs]]
method = "character_vector_graph"

[outputs.params]
filename = "rps_characters.svg"
```

### 5.2 子ノード + ファイル参照

```toml
method = "general_from_team_matchups"
name = "team_matrix"

[refs]
teams = "teams/default.toml"

[children.character_matrix]
method = "monocycle_from_characters"
name = "character_matrix"

[children.character_matrix.refs]
characters = "characters/default.toml"

[params]
use_monocycle_formula = true

[[outputs]]
method = "payoff_directed_graph"

[outputs.params]
filename = "team_matrix.svg"
threshold = 0.0
```

## 6. エラーになる代表ケース

- ルートに `method` がない
- `[[outputs]]` 要素に `method` がない
- 未知の `method` / `outputs[].method`
- 必須 `children` がない（例: `general_from_team_matchups` の `children.character_matrix`）

