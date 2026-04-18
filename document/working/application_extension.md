# アプリケーション層の機能追加ガイド

この文書は、現行 `src/` 実装に対して機能追加するときの最短手順をまとめたものです。  
対象: `src/monocycle_nash/application/` を中心とした拡張（ノード追加・出力追加・入力参照追加）。

## 1. MatrixNode（`method`）を追加する場合

### 追加先

- `src/monocycle_nash/application/matrix_nodes.py`

### 必須実装

1. `MatrixNode` を継承したクラスを追加し、`node_method="..."` をクラス宣言に付ける
2. `@classmethod _from_spec(cls, spec, build_child)` を実装する
3. `build(self, ctx)` を実装する

### 実装時の要点

- `name` と `outputs` フィールドを持たせる（既存ノードと同じ形にする）
- 子ノードを使う場合は `spec.children[...]` を取り、`build_child(...)` で構築する
- 出力定義は `OutputNode.create_all_from_specs(spec.outputs)` で受け取る

### 登録作業の要否

- `MatrixNodeFactory` の編集は不要
- 配列や辞書へのクラス名追記も不要  
  （`MatrixNode.__init_subclass__` が `node_method` を自動登録）

## 2. OutputNode（`outputs[].method`）を追加する場合

### 追加先

- `src/monocycle_nash/application/matrix_nodes.py`

### 必須実装

1. `OutputNode` を継承したクラスを追加し、`output_method="..."` をクラス宣言に付ける
2. `@classmethod _from_output_spec(cls, spec)` を実装する
3. `run(...)-> Path` を実装する

### 登録作業の要否

- 配列や辞書へのクラス名追記は不要  
  （`OutputNode.__init_subclass__` が `output_method` を自動登録）

## 3. `refs` 入力ソースを追加する場合

### 追加先（代表）

- 抽象ポート: `src/monocycle_nash/application/ports.py`
- ノード側の解決呼び出し: `src/monocycle_nash/application/matrix_nodes.py`
- 解決セッション実装: `src/monocycle_nash/application/matrix_config_tree.py`
- 具象実装（TOML）: `src/monocycle_nash/infrastructure/input/`
- CLI配線: `src/monocycle_nash/presentation/cli.py`

### 必須実装

1. 新しい Port インターフェース（抽象メソッド）を定義
2. `NodeResolutionContext` に対応メソッドを追加
3. `_ResolutionSession` で上記メソッドを実装
4. インフラ層に具象 Port 実装を追加
5. CLI の `MatrixConfigTreeResolver(...)` 生成時に Port を注入

### 登録作業の要否

- `src/monocycle_nash/infrastructure/input/__init__.py` の `__all__` への追記は必要
  （import 公開面を維持するため）

## 4. 設定スキーマ（NodeSpec/TOML）を拡張する場合

### 追加先

- DTO: `src/monocycle_nash/application/node_spec.py`
- パーサ: `src/monocycle_nash/infrastructure/input/toml_matrix_config_port.py`
- スナップショット保存: `src/monocycle_nash/infrastructure/output/toml_snapshot_store.py`

### 必須確認

- 新フィールドのデフォルトを `NodeSpec` 側で定義して後方互換を維持する
- TOML→DTO の変換で新フィールドを読み込む
- スナップショット出力に反映されることを確認する

## 5. ドキュメント更新ルール

機能追加で公開仕様が変わる場合は、以下を同時に更新する。

- `README.md`（概要・使い方・method 一覧）
- `document/working/toml_config.md`（入力スキーマ詳細）

## 6. 最低限の確認コマンド

```bash
python -m pytest tests/ -x -q
```
