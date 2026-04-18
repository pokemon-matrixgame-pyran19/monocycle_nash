# 入力読み込み設計

## 概要

実験条件の入力データと実行設定の入力データを分離して管理する。

- 実験条件: `data/<class>/<id>/data.toml`
- 実行設定: `data/setting/*.toml`

どちらも TOML を読み込み、`$ref` を使って設定の一部を別ファイルへ分割できる。

## 参照ドキュメント

- `class_design.md`: 読み込みクラスの責務・公開API・エラーハンドリング
- `usecase_entrypoints.md`: ユースケース別エントリーポイントと入出力フォーマット案
- `data.md`: 入力フォーマットの背景と利用イメージ

## 補足

- `usecase_entrypoints.md` 内の一部参照は旧パス表記を含むため、必要に応じて `initial_design` と `management` 配下の文書へ読み替える
