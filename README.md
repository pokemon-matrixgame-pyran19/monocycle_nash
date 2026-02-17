# 相性が1種類ゲームの均衡計算

利得行列を「パワー + 相性」の形に分けて扱い、相性ベクトルが1種類（2次元ベクトル）で表せるケースを中心に均衡を計算するプロジェクトです。

## セットアップ

`uv` で Python を入れていれば、プロジェクト直下で次を実行するだけで依存関係を準備できます。

```bash
uv sync
```

## 主な使い方

このプロジェクトでは次の3つのエントリポイントを使います。

- 均衡解の計算: `solve_payoff`
- 利得関係の有向グラフ出力: `graph_payoff`
- キャラクターベクトルの可視化: `plot_characters`

どれも `run_config` を指定して実行します。

```bash
uv run solve_payoff --run-config baseline/janken_solve
uv run graph_payoff --run-config baseline/janken_graph
uv run plot_characters --run-config baseline/janken_char_plot
```

### 実行結果

- `solve_payoff`:
  - `results/<run_id>/output/equilibrium.json`
  - `results/<run_id>/output/pure_strategy.json`
  - `results/<run_id>/output/divergence.json`
- `graph_payoff`:
  - `results/<run_id>/output/edge_graph.svg`
- `plot_characters`:
  - `results/<run_id>/output/character_vector.svg`

あわせて実行履歴が `.runmeta/run_history.db` に記録されます（設定で変更可能）。

## 管理システムCLI

本体コマンド（`solve_payoff` / `graph_payoff` / `plot_characters`）を通常実行した場合でも、
実行履歴は管理システムにより自動でDBへ保存されます（既定: `.runmeta/run_history.db`）。

また、`setting.analysis_project` を設定している場合は、runと考察用ディレクトリを結びつける参照リンク生成にも対応しています。
管理システムCLI（`runmeta`）を使うと、これらの実行履歴・project情報の一覧確認や更新、参照の再生成を行えます。

詳細な使い方は次のドキュメントに分離しています。

- [`document/management/runmeta_cli.md`](document/management/runmeta_cli.md)

## 入力ファイルの準備方法

`--run-config` で指定する文字列は、`data/run_config/<name>.toml`（拡張子省略）を指します。  
例: `--run-config baseline/janken_solve` → `data/run_config/baseline/janken_solve.toml`

### 1. run_config を作る

最小構成は次のようになります。

```toml
# data/run_config/<group>/<case>.toml
matrix = "<matrix名>"
setting = "<setting名>"
# graph_payoff / plot_characters を使う場合は graph が必要
# graph = "<graph名>"
```

- `matrix`: `data/matrix/<matrix名>/data.toml` を参照
- `setting`: `data/setting/<setting名>.toml` を参照
- `graph`: `data/graph/<graph名>/data.toml` を参照（`graph_payoff` / `plot_characters` で必須）

### 2. matrix を作る（2通り）

`data/matrix/<matrix名>/data.toml` は、次のどちらか片方で記述します。

#### A. 行列を直接書く

```toml
labels = ["A", "B", "C"]
matrix = [
  [0.0, 1.0, -1.0],
  [-1.0, 0.0, 1.0],
  [1.0, -1.0, 0.0],
]
```

#### B. キャラクター情報から作る

```toml
characters = [
  { label = "A", p = 0.0, v = [ 2.0,  0.0] },
  { label = "B", p = 0.0, v = [-1.0,  1.7] },
  { label = "C", p = 0.0, v = [-1.0, -1.7] },
]
```

制約:

- `matrix` と `characters` は同時に指定できません（どちらか片方のみ）。
- `matrix` は正方行列である必要があります。
- `characters` は `label`（文字列）, `p`（数値）, `v`（長さ2の数値配列）を持つ必要があります。
- `plot_characters` を使う場合は `matrix` ではなく `characters` 入力が必須です。

### 3. graph を作る（可視化系エントリポイントで使用）

`data/graph/<graph名>/data.toml` の例:

#### A. 利得有向グラフ（`graph_payoff`）

```toml
threshold = 0.0
canvas_size = 840
```

- `threshold`（任意）: 辺を描画する下限
- `canvas_size`（任意）: 画像サイズ

#### B. キャラクターベクトル図（`plot_characters`）

```toml
canvas_size = 840
margin = 90
```

- `canvas_size`（任意）: 画像サイズ
- `margin`（任意）: 図の余白

### 4. setting を作る

`data/setting/<setting名>.toml` の例:

```toml
[runmeta]
sqlite_path = ".runmeta/run_history.db"

[output]
base_dir = "results"

[analysis_project]
project_id = "prototype"
project_path = "project_prototype"
```

最低限、`runmeta` と `output` を置いておくと運用しやすいです。

## サンプルデータ

- 均衡計算: `baseline/janken_solve`
- グラフ出力: `baseline/janken_graph`
- キャラクター可視化: `baseline/janken_char_plot`

まずは次を実行すると全体像をつかみやすいです。

```bash
uv run solve_payoff --run-config baseline/janken_solve
uv run graph_payoff --run-config baseline/janken_graph
uv run plot_characters --run-config baseline/janken_char_plot
```

## テスト

```bash
uv run pytest
```

## フォルダ構造

- `src/`: Python ソースコード
- `data/`: 入力ファイル・設定値
- `document/`: 文書
- `tests/`: テストコード（pytest）
- `README.md`: 本ファイル
