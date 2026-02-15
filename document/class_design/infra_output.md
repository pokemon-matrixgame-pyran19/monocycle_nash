# インフラ層（実行結果管理/DB書き込み）設計

## 目的

`document/management/run_result_management_spec.md` で定義した実行結果管理を、`monocycle_nash` のPythonコードから実際に扱えるようにするためのインフラ層クラス設計を定義する。

この設計により、以下を満たす。

- 実行開始時に `runs` へ `INSERT` して `run_id` を採番できる。
- 実行終了時に `runs` を更新し、`result/<run_id>/meta.json` を確定できる。
- `projects` / `runs` の検索・更新・削除をアプリ層から安全に実行できる。

> 結論として、**DB書き込みは可能**。`sqlite3` 標準ライブラリで実装でき、管理仕様の要件（連番採番、状態更新、検索）にそのまま対応できる。

---

## レイヤ構成

```text
Application (usecase / CLI)
  ↓
RunSessionService（実行開始〜終了のユースケース）
  ↓
Repository層（RunsRepository / ProjectsRepository）
  ↓
SQLite層（SQLiteConnectionFactory / Transaction）
  ↓
Filesystem層（RunArtifactStore: result/<run_id>/）
```

- ドメイン計算（利得行列/均衡解）と、永続化（DB/ファイル）を分離する。
- DBはRepositoryに集約し、`sqlite3` の生SQLをユースケース層へ漏らさない。

---

## 推奨配置

```text
src/monocycle_nash/runmeta/
  __init__.py
  models.py            # RunRecord / ProjectRecord
  clock.py             # JSTタイムスタンプ供給
  db.py                # SQLiteConnectionFactory, migrate()
  repositories.py      # RunsRepository, ProjectsRepository
  artifact_store.py    # result/<run_id>/ の入出力
  service.py           # RunSessionService（開始/終了）
  cli.py               # runmetaサブコマンド
```

---


## 管理仕様との対応表

`document/management/run_result_management_spec.md` の必須要件との対応を明確にする。

| 管理仕様の要件 | 本設計での担当クラス/処理 |
|---|---|
| `PRAGMA foreign_keys = ON` を接続ごとに有効化 | `SQLiteConnectionFactory.connect()` |
| `runs` へ `running` でINSERTし `run_id` 採番 | `RunsRepository.create_running(...)` |
| `result/<run_id>/` に `input/output/logs/meta.json` を保持 | `RunArtifactStore` |
| 終了時に `status`/`ended_at`/`updated_at` を更新 | `RunsRepository.finish(...)` + `RunSessionService.finish_*` |
| `created_at` 範囲・`status`・`project_id` で検索 | `RunsRepository.list_runs(...)` |
| CLIで update/delete/list/add-project などを実施 | `runmeta/cli.py` + Repository呼び出し |

この表により、実装時に「どこへ何を書くか」を迷わず追跡できる。

---

## クラス設計

### 1) SQLiteConnectionFactory

**責務**
- `.runmeta/run_history.db` への接続生成。
- 接続ごとに `PRAGMA foreign_keys = ON;` を実行。
- 初期化時にテーブル/インデックス作成（migrate）。

**主要メソッド**
- `connect() -> sqlite3.Connection`
- `migrate(conn) -> None`

```python
class SQLiteConnectionFactory:
    def __init__(self, db_path: Path):
        self._db_path = db_path

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn
```

### 2) RunsRepository

**責務**
- `runs` テーブルへのCRUD。
- 実行開始時の `INSERT(status='running')` と `run_id` 返却。
- 実行終了時の `status/ended_at/updated_at` 更新。

**主要メソッド**
- `create_running(command, git_commit, note, project_id, started_at, created_at, updated_at) -> int`
- `finish(run_id, status, ended_at, updated_at, note=None) -> None`
- `update(run_id, *, status=None, note=None, project_id=None, updated_at) -> None`
- `delete(run_id) -> None`
- `find_by_id(run_id) -> RunRecord | None`
- `list_runs(from_at=None, to_at=None, status=None, project_id=None) -> list[RunRecord]`

### 3) ProjectsRepository

**責務**
- `projects` テーブルへのCRUD。

**主要メソッド**
- `add(project_id, project_path, created_at, note="") -> None`
- `update(project_id, *, project_path=None, note=None) -> None`
- `delete(project_id) -> None`
- `find(project_id) -> ProjectRecord | None`

### 4) RunArtifactStore

**責務**
- `result/<run_id>/` のディレクトリ生成。
- `input/`, `output/`, `logs/`, `meta.json` の書き込み。
- `run_id` からパスを決定（DBに `result_path` を持たせない）。

**主要メソッド**
- `create_run_dir(run_id) -> Path`
- `save_input_snapshot(run_id, src_input_path, merged_input_bytes=None) -> InputSnapshotInfo`
- `write_initial_meta(run_id, payload) -> None`
- `write_final_meta(run_id, payload) -> None`
- `delete_run_dir(run_id) -> None`

### 5) RunSessionService

**責務**
- 「DB先行」フローを1ユースケースとして実行。
- 異常時に `fail` / `killed` を可能な範囲で反映。

**主要メソッド**
- `start(command, project_id=None, note="") -> RunContext`
- `finish_success(ctx) -> None`
- `finish_fail(ctx, reason=None) -> None`

**開始フロー（仕様対応）**
1. DB接続取得・migrate
2. `runs` に running でINSERT → `run_id` 採番
3. `result/<run_id>/` 作成
4. 入力スナップショット保存
5. `meta.json` 初期生成

**終了フロー**
1. 実行結果を `output/` / `logs/` に出力
2. `runs` 更新（status, ended_at, updated_at）
3. `meta.json` 確定保存

---

## データモデル（例）

```python
@dataclass(frozen=True)
class RunRecord:
    run_id: int
    created_at: str
    started_at: str
    ended_at: str | None
    status: Literal["running", "success", "fail", "killed"]
    command: str
    git_commit: str | None
    note: str
    project_id: str | None
    updated_at: str
```

---

## トランザクション方針

- `start()` のDB INSERTは `commit` 後に `run_id` を確定させる。
- ファイル操作とDB更新は分散トランザクションにしない（SQLite + FSのため）。
- 失敗時は以下で整合性を担保：
  - `result/<run_id>/` 作成後に処理失敗 → `runs.status='fail'` 更新を優先。
  - DB更新失敗時はログに残し、再実行可能なCLI（`update-run`）で補正。

---

## 例: 最小ユースケース（疑似コード）

```python
ctx = run_session_service.start(
    command="uv run python -m monocycle_nash ...",
    project_id="analysis-main",
)
try:
    result = execute_main_logic()
    artifact_store.write_output(ctx.run_id, result)
    run_session_service.finish_success(ctx)
except KeyboardInterrupt:
    run_session_service.finish_killed(ctx)
    raise
except Exception:
    run_session_service.finish_fail(ctx)
    raise
```

---

## 既存コードへの接続ポイント

- エントリーポイント `src/main.py` で実行開始時に `RunSessionService.start()` を呼ぶ。
- 既存の計算ドメイン（`matrix`, `solver`, `equilibrium` など）は変更せず、外側にrunmeta層を追加する。
- CLIは `uv run python -m monocycle_nash.runmeta.cli ...` 形式で追加する。

これにより、既存の計算ロジックを壊さずに実行履歴のDB書き込みと検索運用を導入できる。
