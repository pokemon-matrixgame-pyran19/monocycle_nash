# テスト項目

## 概要

テストを「理論予測テスト」と「実装確認テスト」に分類。

| 分類 | 目的 | 項目数 |
|------|------|--------|
| **理論予測テスト** | 数学的な理論値と実装結果の一致を確認 | 最小限 |
| **実装確認テスト** | 実装上必要な振る舞いを確認 | 必要に応じて |

## ディレクトリ構成

```
document/test/
├── index.md                    # このファイル
├── janken.md                   # 元のメモ（参考用）
├── theory/                     # 理論予測テスト
│   ├── index.md               # 理論予測の概要・テストパターン
│   └── test_builder_design.md # テストビルダーの設計
└── implementation/            # 実装確認テスト
    ├── index.md
    ├── matchup.md
    ├── pool.md
    ├── batch_env.md
    └── calc_a.md
```

## 理論予測テスト（theory/）

[theory/index.md](./theory/index.md) を参照。

テストビルダーで各利得行列の理論値を一元管理し、単体テストで入力→処理→出力を検証。

## 実装確認テスト（implementation/）

実装上の振る舞いを確認。

| ファイル | 内容 |
|----------|------|
| [matchup.md](./implementation/matchup.md) | 外積の交換法則 |
| [pool.md](./implementation/pool.md) | 等価判定、convert機能 |
| [batch_env.md](./implementation/batch_env.md) | BatchEnvironmentの一致確認 |
| [calc_a.md](./implementation/calc_a.md) | 順序非依存、内部/外部判定 |

## 分類の判断基準

### 理論予測テスト
- 数学的に計算できる値
- クラス設計の各層（matrix, solver, team, isopower）に対応
- **項目数は最小限に**

### 実装確認テスト
- 実装上必要だが数学的な「正解」がないもの
- 実装の一貫性・対称性を確認
