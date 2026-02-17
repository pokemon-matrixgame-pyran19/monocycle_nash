# runmeta CLI 使い方

`runmeta` は、実行履歴DB（既定: `.runmeta/run_history.db`）を操作する管理用CLIです。  
メインの解析コマンド（`solve_payoff` / `graph_payoff` / `plot_characters`）を補助する用途を想定しています。

## 起動方法

```bash
uv run runmeta --help
```

DBの場所を切り替える場合は `--db-path` を付けます。

```bash
uv run runmeta --db-path .runmeta/run_history.db list-runs
```

## サブコマンド一覧

- `list-runs`: 実行履歴を一覧表示
- `update-run`: run の `status` / `note` / `project_id` を更新
- `delete-run`: run を削除（`--with-files` で出力フォルダも削除）
- `list-projects`: プロジェクト一覧を表示
- `add-project`: プロジェクトを追加
- `update-project`: プロジェクト情報を更新
- `delete-project`: プロジェクトを削除
- `regenerate-project-refs`: project に紐づく run の参照リンクを再生成

## よく使う例

### 実行履歴の確認

```bash
uv run runmeta list-runs
uv run runmeta list-runs --status success
uv run runmeta list-runs --project-id prototype
```

### run の更新・削除

```bash
uv run runmeta update-run --run-id 12 --status success --note "収束条件を見直し"
uv run runmeta update-run --run-id 12 --project-id prototype
uv run runmeta delete-run --run-id 12
uv run runmeta delete-run --run-id 12 --with-files
```

### project の管理

```bash
uv run runmeta add-project --project-id prototype --project-path project_prototype --note "試験運用"
uv run runmeta update-project --project-id prototype --note "メモ更新"
uv run runmeta list-projects
uv run runmeta delete-project --project-id prototype
```

### 参照リンクの再生成

```bash
uv run runmeta regenerate-project-refs --project-id prototype --result-base-dir results
```

## 補足

- `runmeta` 実行時は必要に応じてマイグレーションが走り、DBスキーマを整えます。
- `list-runs` / `list-projects` の時刻表示は JST（`YYYY-MM-DD HH:MM`）です。
- `update-run` は `--status` `--note` `--project-id` のいずれか1つ以上が必須です。
- `update-project` は `--project-path` か `--note` のどちらかが必須です。
