# monocycle_nash

相性が1種類（monocycle）で表せる零和ゲームを中心に、利得行列の分析ロジックを扱うプロジェクトです。

## 現在の状態（重要）

アプリケーション層以降を作り直し中です。
現行の CLI エントリーポイント（`python -m monocycle_nash.main`）から、
「設定読込 → ノード生成 → 解決 → 出力保存」を実行できます。

- 現行の実装対象: `src/monocycle_nash/domain/` を中心としたドメインロジック
- 再構築中の層: application / infrastructure / presentation

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
- README補助ドキュメントを `document/readme/` で確認する

## 最小実行手順（現行）

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

2) CLI から実行:

```bash
python -m monocycle_nash.main /absolute/path/to/rps_inline.toml
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
runner = "..."                  # 任意（同名 runner で最終集約）
[outputs.params]                # 任意
```

### `method` 一覧（行列構築ノード）

- `general_from_raw`（`params.matrix` 必須）
- `monocycle_from_characters`（`params.characters` または `refs.characters`）
- `general_from_teams_payoff`（`params.team_payoff` 必須、`params.teams` または `refs.teams`）
- `general_from_team_matchups`（`children.character_matrix` 必須、`params.teams` または `refs.teams`）
- `random_skew_symmetric`（`params.size` 必須）
- `approx_monocycle_to_general`（`children.source` 必須）
- `approx_dominant_eigenpair`（`children.source` 必須）
- `approx_equilibrium_preserving`（`children.source` 必須）

### `outputs[].method` 一覧（OutputNode）

- `outputs[].runner`
  - 任意。同じ値を指定した出力は同一 runner に送信され、**全ノード解決後に runner が1回だけ**実行されます。
  - runner 内では `OutputNode.resolve_batch_key()` ごとに emit がまとめられ、`OutputNode.execute_emissions(...)` が呼ばれます。
  - 既定実装は従来どおり emit ごとに `execute(...)` を呼ぶため後方互換です。
  - 省略時は後方互換の単体実行（出力ごとに独立 runner）です。
- `payoff_directed_graph`
  - 任意パラメータ: `filename`, `threshold`, `canvas_size`
- `character_vector_graph`
  - 任意パラメータ: `filename`, `canvas_size`, `margin`
  - `characters` domain を持つノードにのみ利用可
- `equilibrium`
  - 任意パラメータ: `filename`
  - 均衡解を `strategies = [{ id, probability }, ...]` 形式のTOMLで出力

### 出力先パス仕様

`FileSystemOutputPathPort` は以下へ出力します。

```text
result/<run_id>/<node_path...>/<output_method>/<filename>
```

## 開発者向け: 機能追加ガイド

機能追加時の実装手順は以下を参照してください。

- [`document/readme/application_extension.md`](document/readme/application_extension.md)

このガイドでは次を整理しています。

- どのファイルにクラスを追加するか
- どのメソッド実装が必須か
- クラス名の手動登録（配列や辞書追記）が必要かどうか
  - `MatrixNode` / `OutputNode` は自動登録のため、通常は追記不要

## サンプル設定ファイル

`data/` 直下に実行可能なサンプルを配置しています。

- `data/rps_inline.toml`
- `data/random_5.toml`
- `data/team_matchups_inline.toml`
- `data/approx_equilibrium_preserving.toml`

## TOML仕様の詳細

詳細は [`document/readme/toml_config.md`](document/readme/toml_config.md) を参照してください。

## ドキュメントの見方

- `document/readme/`: READMEの補助ドキュメント
- `document/working/`: 現在進行中の作業メモ（通常は `index.md` のみ）
- `document/class_design/`: 旧仕様を含む**アーカイブ**
- `document/theory/`: 理論背景
- `document/test/`, `document/test_strategy.md`: テスト方針・設計メモ
- `document/experiment/`: 実験計画・実験ログ

> 旧CLIの具体的な使い方（`solve_payoff` など feature 実行）は、再構築前仕様としてアーカイブ側に残しています。

## フォルダ構造

- `src/`: 現行ソースコード
- `data/`: 入力データ・設定サンプル
- `document/`: 設計・仕様・理論・実験ドキュメント
- `tests/`: テストコード
