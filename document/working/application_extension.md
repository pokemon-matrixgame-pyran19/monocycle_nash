# 機能追加ガイド

このガイドでは、新しい行列生成方式・出力形式・ファイル入力形式などを追加するときの手順をまとめています。

## 1. 新しい行列生成方式（`method`）を追加したい

TOML 設定の `method = "..."` で指定できる新しい行列の作り方を追加するケースです。

### 変更するファイル

- `src/monocycle_nash/application/matrix_nodes.py`

### 実装手順

1. `MatrixNode` を継承したクラスを作り、クラス宣言に `node_method="新しいmethod名"` を付ける
2. `_from_spec(cls, spec, build_child)` classmethod を実装する（TOML から読み込んだ設定 `spec` をもとにインスタンスを組み立てる）
3. `build(self, ctx)` メソッドを実装する（行列の計算ロジックを書き、`PayoffMatrix` を返す）

### 実装上の要点

- `name` と `outputs` フィールドを持たせる（既存ノードと同じ形にする）
- 子ノードに依存する場合は `spec.children["key"]` で取り出し、`build_child(...)` でノードを組み立てる
- 出力定義は `OutputNode.create_all_from_specs(spec.outputs)` で受け取る

### クラス登録の作業は不要

他ファイルへのクラス名の追記は不要です。クラスを定義するだけで `method` として自動的に認識されます。

### ドキュメント更新

公開仕様が変わるため、以下も更新してください。

- `README.md`（`method` 一覧）
- `document/working/toml_config.md`（スキーマ詳細）

## 2. 新しい出力形式（`outputs[].method`）を追加したい

TOML 設定の `[[outputs]]` で指定できる新しい出力の種類を追加するケースです。

### 変更するファイル

- `src/monocycle_nash/application/matrix_nodes.py`

### 実装手順

1. `OutputNode` を継承したクラスを作り、クラス宣言に `output_method="新しいmethod名"` を付ける
2. `_from_output_spec(cls, spec)` classmethod を実装する（TOML の `outputs.params` をもとにインスタンスを組み立てる）
3. `run(self, *, output_path_port, run_id, node_path, matrix) -> Path` メソッドを実装する（出力ファイルを生成してパスを返す）

### クラス登録の作業は不要

他ファイルへのクラス名の追記は不要です。クラスを定義するだけで `outputs[].method` として自動的に認識されます。

### ドキュメント更新

- `README.md`（`outputs[].method` 一覧）
- `document/working/toml_config.md`（スキーマ詳細）

## 3. ファイル参照による新しい入力形式を追加したい

`refs.characters` / `refs.teams` のように、外部ファイルからデータを読み込む入力形式を新たに追加するケースです。現在の `refs.characters` / `refs.teams` に倣って実装します。

### 変更するファイル

1. `src/monocycle_nash/application/ports.py` — 読み込み処理を定義するインターフェースを追加する
2. `src/monocycle_nash/application/matrix_nodes.py` — ファイル参照入力を使うノードに読み込み呼び出しを追加する
3. `src/monocycle_nash/application/matrix_config_tree.py` — セッション内で新しい読み込みを実行できるようにする
4. `src/monocycle_nash/infrastructure/input/` — TOML を読み込む具体的な処理を追加する（`toml_ref_file_ports.py` に追記するか新規ファイルを作る）
5. `src/monocycle_nash/infrastructure/input/__init__.py` — 追加したクラスを `__all__` に載せる
6. `src/monocycle_nash/presentation/cli.py` — CLIの実行時に新しい読み込み処理が使われるよう配線する

### 実装手順の概要

1. `ports.py` に抽象クラスを追加して読み込みメソッドを定義する
2. ノード側で `NodeResolutionContext` を通じた呼び出しを追加する
3. `_ResolutionSession` に具体的な呼び出し処理を実装する
4. TOML読み込みの具体処理を `infrastructure/input/` に実装する
5. `cli.py` の `MatrixConfigTreeResolver(...)` 生成時に新しい読み込み処理を渡す

## 4. TOML 設定スキーマを拡張したい

TOML ファイルに新しいパラメータやフィールドを追加するケースです。

### 変更するファイル

1. `src/monocycle_nash/application/node_spec.py` — 設定データ構造（`NodeSpec`）に新しいフィールドを追加する
2. `src/monocycle_nash/infrastructure/input/toml_matrix_config_port.py` — TOML からフィールドを読み取る処理を追加する
3. `src/monocycle_nash/infrastructure/output/toml_snapshot_store.py` — スナップショットへの反映を確認・更新する

### 実装上の要点

- 新フィールドにはデフォルト値を設定して既存の TOML ファイルへの後方互換を維持する

### ドキュメント更新

- `README.md`
- `document/working/toml_config.md`

## 5. 動作確認コマンド

```bash
python -m pytest tests/ -x -q
```
