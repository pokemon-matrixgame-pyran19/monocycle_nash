# ユースケース別エントリーポイントと入出力フォーマット案

## 目的

`document/usecase.md` のユースケース (1)〜(3) を実装する前提で、以下を先に固定する。

- エントリーポイントの分割方針
- `data/` 配下の入力 TOML の配置とスキーマ
- 実行時に参照する「ルート設定ファイル」の方式
- 出力ファイルと入力スナップショットの配置規約

将来ユースケース (4)(5) はこの文書では対象外とする。

## 前提

- 後方互換性は考慮しない（現時点で仕様を整理してよい）。
- 入力は `document/input_loader/class_design.md` のローダー規約（`$ref` 解決）に従う。
- 実験条件（利得行列・グラフ条件）と環境設定（runmeta / 出力先など）は分離する。
- 実行結果管理は `document/management/run_result_management_spec.md` と整合を取る。

---

## 1. エントリーポイント案

ユースケースごとに実行入口を分離する。

1. **graph_payoff**: 利得行列の有向グラフを出力（usecase 1）
2. **solve_payoff**: 利得行列の均衡・純粋戦略・乖離度を出力（usecase 2）
3. **plot_characters**: キャラクター配列の相性ベクトル散布図を出力（usecase 3）

### 実行イメージ（実装時）

```bash
uv run python -m monocycle_nash.entrypoints.graph_payoff --run-config baseline/rps3_graph
uv run python -m monocycle_nash.entrypoints.solve_payoff --run-config baseline/rps3_solve
uv run python -m monocycle_nash.entrypoints.plot_characters --run-config baseline/char_plot
```

CLI には個別の `--matrix` / `--graph` / `--setting` を並べず、`--run-config` で 1 つの設定を指定する。

---

## 2. data 配下のディレクトリ規約

```text
data/
  matrix/
    <id>/
      data.toml
      ... ($ref で分割した下位toml)
  graph/
    payoff/
      <id>/
        data.toml
    character/
      <id>/
        data.toml
  setting/
    <name>.toml
    ... ($ref で分割したtoml)
  run_config/
    <group>/
      <name>.toml
```

- `matrix`: 利得行列生成入力（直接行列 or キャラクター）
- `graph`: 描画時パラメータ
- `setting`: 環境設定（DB, 出力先, 考察プロジェクトなど）
- `run_config`: 実行時に参照するルート設定（どの matrix/graph/setting を使うか）

---

## 3. 利得行列入力 (`data/matrix/<id>/data.toml`)

トップレベルは `matrix` か `characters` のどちらかを持つ。
必要になった時点で下位セクションを追加してよい（現時点で固定しない）。

## 3.1 直接行列指定

```toml
# data/matrix/rps3/data.toml
matrix = [
  [0.0, 1.0, -1.0],
  [-1.0, 0.0, 1.0],
  [1.0, -1.0, 0.0],
]
```

### バリデーション

- 2 次元数値配列。
- 正方行列。
- （単相性モデル前提なら）交代行列制約は別途チェック。

## 3.2 キャラクター指定（単相性モデル）

```toml
# data/matrix/char_example/data.toml
characters = [
  { label = "A", p = 1.0, v = [1.0, 0.0] },
  { label = "B", p = 1.2, v = [0.0, 1.0] },
  { label = "C", p = 0.8, v = [-1.0, -1.0] },
]
```

### バリデーション

- 1 件以上。
- 各要素は `label`, `p`, `v=[x,y]` を持つ。
- `label` の重複禁止。

## 3.3 排他ルール

- `matrix` と `characters` の同時指定は禁止。
- どちらも無い場合はエラー。

---

## 4. グラフ設定入力 (`data/graph/.../data.toml`)

描画系ユースケースで利用する値を TOML 化する。

## 4.1 利得行列グラフ (`data/graph/payoff/<id>/data.toml`)

```toml
threshold = 0.0
canvas_size = 840
```

- `threshold`: 辺を描画する最小利得（`value > threshold` を描画）。
- `canvas_size`: SVG の縦横サイズ。

## 4.2 キャラクター散布図 (`data/graph/character/<id>/data.toml`)

```toml
canvas_size = 840
margin = 90
```

- `canvas_size`: SVG の縦横サイズ。
- `margin`: プロット余白。

---

## 5. 一般設定 (`data/setting/<name>.toml`)

管理システム・出力配置・考察プロジェクト連携を settings にまとめる。

```toml
[runmeta]
sqlite_path = ".runmeta/run_history.db"

[output]
base_dir = "result"

[analysis_project]
project_id = "analysis-main"
project_path = "C:\\analysis\\main"
```

- `runmeta.sqlite_path`: run履歴 DB。
- `output.base_dir`: 実行結果ルート（`result` を想定）。
- `analysis_project.project_id`: run を紐づける考察プロジェクトID。
- `analysis_project.project_path`: 考察プロジェクト実体パス。

`analysis_project` は `run_result_management_spec.md` の `projects` テーブル登録・関連付け時に使用する。

---

## 6. ルート設定ファイル (`data/run_config/<group>/<name>.toml`)

各実行で使う `matrix/graph/setting` を 1 ファイルで指定する。

```toml
# data/run_config/baseline/rps3_graph.toml
matrix = "rps3"
graph = "payoff/default"
setting = "local"
```

### 読み取りルール

- `matrix`: `data/matrix/<matrix>/data.toml`
- `graph`: `data/graph/<graph>/data.toml`
- `setting`: `data/setting/<setting>.toml`

`solve_payoff` のようにグラフ設定が不要なエントリーポイントでは、`graph` を省略可能とする。

---

## 7. 出力ファイル規約（run_id 基準）

`run_result_management_spec.md` に合わせ、`output.base_dir/<run_id>/` を実行単位の固定保存先とする。

```text
<output.base_dir>/
  <run_id>/
    input/
      matrix.toml
      graph.toml          # graph不要のrunでは省略可
      setting.toml
    output/
      ... エントリーポイントごとの成果物 ...
    logs/
      stdout.log
      stderr.log
    meta.json
```

### 入力スナップショット方針

- `input/` 配下には **3種類（matrix / graph / setting）** を最終的に保存する。
- `$ref` で分割されていても、実行時には解決・合成した TOML として保存する。
- グラフ設定が不要な run では `graph.toml` を省略してよい。

### エントリーポイント別出力例（`output/` 配下）

- `graph_payoff`: `edge_graph.svg`
- `solve_payoff`: `equilibrium.json`, `pure_strategy.json`, `divergence.json`
- `plot_characters`: `character_vector.svg`

エントリーポイント種別はフォルダ名ではなく、runmeta（`runs.command` など）で管理・フィルタする。

---

## 8. ユースケース対応表

- usecase 1（利得行列グラフ）
  - 入力: `run_config -> matrix + graph(payoff) + setting`
  - 出力: `<base_dir>/<run_id>/output/edge_graph.svg`
- usecase 2（均衡計算）
  - 入力: `run_config -> matrix + setting`（必要なら graph は省略）
  - 出力: `<base_dir>/<run_id>/output/*.json`
- usecase 3（キャラクター散布図）
  - 入力: `run_config -> matrix(=characters) + graph(character) + setting`
  - 出力: `<base_dir>/<run_id>/output/character_vector.svg`

---

## 9. 実装タスクへの受け渡しメモ

次タスクでは以下を実装対象にする。

1. エントリーポイント 3 本の追加。
2. `run_config` から `matrix` / `graph` / `setting` を解決してロード。
3. 入力バリデーション（排他・必須・型）。
4. `output.base_dir/<run_id>/` の作成。
5. `input/` に解決済み `matrix.toml` / `graph.toml` / `setting.toml` を保存。
6. `setting.analysis_project.*` を runmeta 連携（project 紐づけ）に利用。
