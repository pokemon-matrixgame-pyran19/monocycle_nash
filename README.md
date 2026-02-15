# 相性が１種類ゲームの均衡計算

利得行列をパワーと相性に分割し、相性部分が1種類の系について計算する。



### 実行方法
以下のように実行するのを想定。コードもそれで動くように書く。

本体の実行
`uv run src\xxx.py`

テストの実行
`uv run pytest`

## フォルダ構造
src/      : pythonソースコード
data/     : 入力ファイル・固有の定数・設定値など
document/ : 文書. 内容ごとにフォルダを作り、フォルダ内の役割はそのフォルダのindex.mdに記載
tests/    : テスト用のコード置き場. pytestを使用

README    : 本ファイル。プロジェクト全体の概要等。

## インストール
uv使ったpython環境を想定. `git clone https://~~`でリポジトリをダウンロードして`uv sync`でインストール完了する。

## Run metadata CLI
`python -m monocycle_nash.runmeta.cli` で実行できます（`pyproject.toml` の `runmeta` エントリポイントからも実行可）。

```bash
# プロジェクト追加
uv run python -m monocycle_nash.runmeta.cli add-project --name baseline

# run更新
uv run python -m monocycle_nash.runmeta.cli update-run --run-id 1 --status success

# run一覧（任意フィルタ）
uv run python -m monocycle_nash.runmeta.cli list-runs --from 2026-01-01T00:00:00+09:00 --status success
```
