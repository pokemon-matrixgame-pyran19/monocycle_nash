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

## 今できること

- ドメイン層のコードを import して利用・検証する
- pytest で既存テストを実行する
- 再構築仕様を `document/working/` で確認する

## テスト

```bash
uv run pytest
```

## TOML設定ファイル（現行仕様）

現行の `TomlMatrixConfigPort` が読み込む TOML 設定の書き方を整理しました。

- 要点: ルートで `method` が必須、`params` / `refs` / `children` / `outputs` で設定を構成
- 詳細仕様: [`document/working/toml_config.md`](document/working/toml_config.md)

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
