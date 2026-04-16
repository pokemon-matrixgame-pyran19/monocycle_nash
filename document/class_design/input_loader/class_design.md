# 入力読み込みクラス設計

## 設計方針

`data.md` の要件を実装しつつ、次の2用途を分離する。

1. 実験条件の読み込み
2. 実行環境設定の読み込み

共通ロジックである `$ref` 解決は汎用クラスへ集約し、用途別クラスはパス解決だけを担当する。

## ディレクトリ規約

### 実験条件

- ルート: `data/<class>/<id>/`
- 入口ファイル: `data.toml`

`$ref` は以下の順で解決する:

1. `<現在ファイルのディレクトリ>/<現在のキー名>/<ref>.toml`
2. `<現在ファイルのディレクトリ>/<ref>.toml`

`$ref` に拡張子やパス区切りが含まれる場合は、そのまま相対パスとして扱う。

### 実行設定

- ルート: `data/setting/`
- 入口ファイル: `<name>.toml` または相対パス指定

## クラス構成

### `TomlTreeLoader`

`$ref` 再帰解決を担当する中核クラス。

- `load(root_file: Path | str) -> dict[str, Any]`
  - TOML を読み込んで木構造を返す
- `load_toml(file_path: Path) -> dict[str, Any]`
  - 1ファイルのTOML読み込み
- `_resolve_node(node, base_dir, current_key, visited)`
  - dict/list/scalar を再帰処理し、`$ref` ノードを置換

#### エラー方針

- 参照先が見つからない: `FileNotFoundError`
- `$ref` ループ検出: `ValueError`
- `$ref` ノードに `$ref` 以外のキーが混在: `ValueError`

### `ExperimentDataLoader`

実験条件フォルダ規約を解決して `TomlTreeLoader` へ委譲する。

- `__init__(base_dir: Path | str = "data", entry_file: str = "data.toml")`
- `load(class_name: str, identifier: str) -> dict[str, Any]`
- `load_from_path(relative_path: Path | str) -> dict[str, Any]`

### `SettingDataLoader`

`data/setting` の設定読み込みを担当。

- `__init__(base_dir: Path | str = "data/setting")`
- `load(name: str) -> dict[str, Any]`
  - `name` が拡張子なしなら `<name>.toml` を読む
- `load_file(relative_file: Path | str) -> dict[str, Any]`

## 依存関係

```text
ExperimentDataLoader ----\
                         +--> TomlTreeLoader
SettingDataLoader -------/
```

用途別ローダー間の依存は持たない。

## テスト観点

1. 単一ファイル読み込み
2. ネストした `$ref` の再帰解決
3. `$ref` のキー名ベース解決（`<key>/<ref>.toml`）
4. 参照ファイル未存在時の例外
5. `$ref` 循環参照の検出
