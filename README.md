# monocycle_nash

相性が1種類（monocycle）で表せる零和ゲームを中心に、利得行列の分析ロジックを扱うプロジェクトです。

## 現在の状態（重要）

アプリケーション層以降を作り直し中です。
現行の `main` エントリーポイント（`uv run main`）は未実装で、実行すると `NotImplementedError` になります。

- 現行の実装対象: `src/monocycle_nash/domain/` を中心としたドメインロジック
- 再構築中の層: application / infrastructure / presentation
- 旧実装参照先: `old/src/monocycle_nash/`

## セットアップ

```bash
uv sync
```

`uv` が使えない環境では、代替として以下でも動作します。

```bash
python -m pip install -e .
```

## テスト実行

```bash
python -m pytest tests/ -x -q
```

## 今できること

- ドメイン層のコードを import して利用・検証する
- `TomlMatrixConfigPort` + `MatrixNodeFactory` + `MatrixConfigTreeResolver` を直接呼び出して設定解決する
- pytest で既存テストを実行する
- 再構築仕様を `document/working/` で確認する

## 最小実行手順（現行）

`main` が未実装のため、現時点では Python から直接呼び出します。

1) 例として `rps_inline.toml` を作成:

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
method = "payoff_directed_graph"

[outputs.params]
filename = "rps.svg"
```

2) 設定を解決して出力を生成:

`python - <<'PY' ... PY` は「その場で Python スクリプトを実行する」書き方です。
必要なら同内容を `run_example.py` に保存して `python run_example.py` でも実行できます。

```bash
python - <<'PY'
from monocycle_nash.application.matrix_config_tree import MatrixConfigTree, MatrixConfigTreeResolver
from monocycle_nash.application.matrix_node_factory import MatrixNodeFactory
from monocycle_nash.infrastructure.input.toml_matrix_config_port import TomlMatrixConfigPort
from monocycle_nash.infrastructure.output import FileSystemOutputPathPort

spec = TomlMatrixConfigPort().load_node_spec("rps_inline.toml")
root = MatrixNodeFactory().build(spec)
result = MatrixConfigTreeResolver(
    output_path_port=FileSystemOutputPathPort(result_base_dir="result")
).resolve(MatrixConfigTree(root=root))

print("run_id:", result.run_id)
print("matrix shape:", result.root.matrix.shape)
print("outputs:", [str(o.path) for o in result.outputs])
PY
```

## 入力仕様（現行）

設定読み込みは `src/monocycle_nash/infrastructure/input/toml_matrix_config_port.py` の
`TomlMatrixConfigPort` が担当します。

- `load_node_spec(config_id)` の `config_id`:
  - 絶対パス（拡張子あり/なし）
  - `data_dir` からの相対パス（拡張子なしなら `.toml` 補完）
- ルートに `method` は必須
- `name` 省略時は `"root"`

トップレベルスキーマ:

```toml
method = "..."                  # 必須
name = "..."                    # 任意
[params]                        # 任意
[refs]                          # 任意
[children.<key>]                # 任意（再帰）
method = "..."
[[outputs]]                     # 任意
method = "..."                  # outputsでは必須
[outputs.params]                # 任意
```

### `method` 一覧（MatrixNode）

- `general_from_raw`（`params.matrix` 必須）
- `monocycle_from_characters`（`params.characters` または `refs.characters`）
- `general_from_teams_payoff`（`params.team_payoff` 必須、`params.teams` または `refs.teams`）
- `general_from_team_matchups`（`children.character_matrix` 必須、`params.teams` または `refs.teams`）
- `random_skew_symmetric`（`params.size` 必須）
- `approx_monocycle_to_general`（`children.source` 必須）
- `approx_dominant_eigenpair`（`children.source` 必須）
- `approx_equilibrium_preserving`（`children.source` 必須）

### `outputs[].method` 一覧（OutputNode）

- `payoff_directed_graph`
  - 任意パラメータ: `filename`, `threshold`, `canvas_size`
- `character_vector_graph`
  - 任意パラメータ: `filename`, `canvas_size`, `margin`
  - `characters` 属性を持つ行列ノードにのみ利用可

### 出力先パス仕様

`FileSystemOutputPathPort` は以下へ出力します。

```text
result/<run_id>/<node_path...>/<output_method>/<filename>
```

## 現時点で不足している機能・入力ファイル

最低限の実行可否に関わる現状の不足点です。

1. エントリーポイント未実装
   - `src/monocycle_nash/main.py` は `NotImplementedError`。
   - CLI から「設定読込→ノード生成→解決→出力保存」を一気通貫で実行できません。

2. `refs` 入力を読むインフラ実装が未提供
   - `CharacterListFilePort` / `TeamListFilePort` は抽象ポートのみで、現行 `src/` に具象実装がありません。
   - そのため、`refs.characters` / `refs.teams` を使う設定はそのままでは実行できません。
   - 現時点で確実に動かすには、`params.characters` / `params.teams` のインライン入力を使ってください。

3. 現行 NodeSpec 形式のサンプル入力が `data/` に不足
   - 既存の `data/run_config/*.toml` や `data/matrix/*/data.toml` は、`old/src/monocycle_nash` の旧CLI運用（feature指定の実行方式）向けサンプルです。
   - 現行の `TomlMatrixConfigPort` が期待する「ルートに `method` を持つ NodeSpec 形式」とは互換ではありません。
   - 現行実装を試す場合は、上記の最小例のような `method` ルート形式の TOML を新規作成してください。

## TOML仕様の詳細

詳細は [`document/working/toml_config.md`](document/working/toml_config.md) を参照してください。

## ドキュメントの見方

- `document/working/`: **現行の作り直し対象**に関する仕様・方針
- `document/class_design/`: 旧仕様を含む**アーカイブ**
- `document/theory/`: 理論背景
- `document/test/`, `document/test_strategy.md`: テスト方針・設計メモ
- `document/experiment/`: 実験計画・実験ログ

> 旧CLIの具体的な使い方（`solve_payoff` など feature 実行）は、再構築前仕様としてアーカイブ側に残しています。

## フォルダ構造

- `src/`: 現行ソースコード
- `old/`: 旧実装コード（参照用）
- `data/`: 入力データ・設定サンプル
- `document/`: 設計・仕様・理論・実験ドキュメント
- `tests/`: テストコード
