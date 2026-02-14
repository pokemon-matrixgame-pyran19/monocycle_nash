# 実行結果管理仕様

## 1. 目的

本仕様は、実行ごとの入力・出力・メタ情報を固定保存し、後から検索・参照しやすくするための管理方式を定義する。

- 実行時点の入力を固定保存する（作業中の入力フォルダ変更の影響を受けない）。
- 実行履歴を一覧管理し、状態・期間・考察プロジェクト単位で検索できる。
- 実行フォルダ内に自己完結したメタ情報を保持する。
- 最低限の運用CLI（更新・削除）を提供する。

---

## 2. 用語

- **run**: 1回のプログラム実行単位。
- **run_id**: runを一意に識別する連番ID。
- **考察プロジェクト（project）**: 実験結果を分析するプロジェクト単位。

---

## 3. ディレクトリ構成

プロジェクト直下に `result/` を作成し、実行ごとにサブフォルダを作成する。

```text
<project-root>/
  result/
    <run_id>/
      input/
        ... 各種入力結果 ...
      output/
        ... 各種実行結果 ...
      logs/
        stdout.log
        stderr.log
      meta.json
  .runmeta/
    run_history.db
```

### 3.1 `result/<run_id>/` の必須要素

- `input/`
  - 実行に使用した入力の固定コピー群。
  - 入力分割設定がある場合は、**実行時に統合した1ファイル**もここに保存する。
- `output/`
  - 実行結果ファイル群。
- `logs/`
  - 標準出力/標準エラーを保存。
- `meta.json`
  - runのメタ情報。

### 3.2 DB配置

- SQLiteは `.runmeta/run_history.db` に配置する。

---

## 4. run_id仕様（連番）

### 4.1 形式

- `run_id` は整数連番（1, 2, 3, ...）とする。
- `runs` テーブルの主キー `run_id INTEGER PRIMARY KEY AUTOINCREMENT` をそのまま使用する。

### 4.2 フォルダ名

- 実行フォルダは `result/<run_id>/` とする（例: `result/42/`）。
- 連番IDから一意に導出できるため、DBに `result_path` は保持しない。

---

## 5. SQLiteスキーマ

時刻はすべて **日本時間（UTC+9）** のISO-8601文字列で保持する。

### 5.0 SQLite設定（必須）

SQLiteはデフォルトで外部キー制約が無効のため、接続ごとに以下を実行して有効化する。

```sql
PRAGMA foreign_keys = ON;
```

### 5.1 projectsテーブル

```sql
CREATE TABLE IF NOT EXISTS projects (
  project_id TEXT PRIMARY KEY,
  project_path TEXT NOT NULL,        -- Windows想定の絶対パス
  created_at TEXT NOT NULL,          -- ISO-8601 (+09:00)
  note TEXT DEFAULT ''
);
```

### 5.2 runsテーブル

```sql
CREATE TABLE IF NOT EXISTS runs (
  run_id INTEGER PRIMARY KEY AUTOINCREMENT,
  created_at TEXT NOT NULL,          -- ISO-8601 (+09:00)
  started_at TEXT NOT NULL,          -- ISO-8601 (+09:00)
  ended_at TEXT,                     -- ISO-8601 (+09:00)
  status TEXT NOT NULL CHECK(status IN ('running','success','fail','killed')),
  command TEXT NOT NULL,             -- 実行コマンド or エントリポイント
  git_commit TEXT,
  note TEXT DEFAULT '',
  project_id TEXT,
  updated_at TEXT NOT NULL,          -- ISO-8601 (+09:00)
  FOREIGN KEY(project_id) REFERENCES projects(project_id)
    ON UPDATE CASCADE ON DELETE SET NULL
);
```

### 5.3 推奨インデックス

```sql
CREATE INDEX IF NOT EXISTS idx_runs_created_at ON runs(created_at);
CREATE INDEX IF NOT EXISTS idx_runs_status ON runs(status);
CREATE INDEX IF NOT EXISTS idx_runs_project_id ON runs(project_id);
```

---

## 6. メタ情報ファイル（meta.json）

`result/<run_id>/meta.json` に以下を保持する。

```json
{
  "run_id": 42,
  "created_at": "2026-02-14T19:30:12+09:00",
  "started_at": "2026-02-14T19:30:12+09:00",
  "ended_at": "2026-02-14T19:31:02+09:00",
  "status": "success",
  "command": "uv run python -m monocycle_nash ...",
  "git_commit": "abc123...",
  "note": "baseline",
  "project_id": "analysis-main",
  "project_path": "C:\\analysis\\main",
  "input": {
    "source_path": "input/",
    "split_config_applied": true,
    "stored_file": "input/merged_input.toml",
    "checksum_sha256": "..."
  }
}
```

---

## 7. 実行時フロー（DB先行）

連番採番のため、実行開始時はDB操作を先に行う。

1. `.runmeta/run_history.db` を開く。
2. `runs` に `status='running'` でINSERTし、`run_id` を採番。
3. `result/<run_id>/` を作成。
4. 入力を読み込み、`input/` に実行時入力の固定コピーを保存（分割設定がある場合は統合後ファイルも保存）。
5. `meta.json` 初期生成。
6. 実処理実行、`output/` と `logs/` を書き込み。
7. 終了時に `status` を `success/fail/killed` へ更新。
8. `ended_at` / `updated_at` を更新し、`meta.json` を確定保存。

異常終了時も可能な範囲で `fail` または `killed` を記録する。

---

## 8. 検索要件

以下で検索可能とする。

- `created_at` 範囲
- `status`
- `project_id`

### 8.1 代表クエリ

```sql
SELECT * FROM runs
WHERE created_at BETWEEN :from AND :to
  AND status = :status
ORDER BY created_at DESC;

SELECT * FROM runs
WHERE project_id = :project_id
ORDER BY created_at DESC;
```

---

## 9. CLI仕様（簡易版）

コマンド例（`uv run python -m monocycle_nash.runmeta ...` 想定）:

- `runmeta update-run --run-id <id> [--status ...] [--note ...] [--project-id ...]`
- `runmeta delete-run --run-id <id> [--with-files]`
- `runmeta add-project --project-id <pid> --project-path <path> [--note ...]`
- `runmeta update-project --project-id <pid> [--project-path <path>] [--note ...]`
- `runmeta delete-project --project-id <pid>`
- `runmeta list-runs [--from ...] [--to ...] [--status ...] [--project-id ...]`

### 9.1 CLI運用ルール

- `run_id` / `project_id` 未存在時は非0終了。

---

## 10. 考察プロジェクトとの参照連携（Windows想定）

run実行時または `project_id` 紐づけ更新時に、考察プロジェクト側に実験結果フォルダ参照を自動配置する。

### 10.1 配置先

```text
<project_path>\experiment_refs\<run_id>
```

### 10.2 参照方式

Windows想定のため以下順で試行する。

1. NTFSシンボリックリンク（作成可能環境）
2. テキスト参照ファイル `experiment_refs\<run_id>.txt`

`<run_id>.txt` には以下を記録する。

- `result\<run_id>` の絶対パス
- 作成日時（+09:00）
- runのステータス

---

## 11. 受け入れ基準

- 実行ごとに `result/<run_id>/` が作成され、`input/` に実行時入力の固定コピーが保存される。
- `runs` に自動登録され、終了時に `status` が確定する。
- `project_id` 指定でrun検索できる。
- `runmeta delete-run --with-files` でDB/フォルダ両方が削除される。
- `meta.json` に開始/終了時刻と主要属性が保存される。

