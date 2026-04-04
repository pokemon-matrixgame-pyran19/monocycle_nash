# team_strict_spectrum 16キャラ×50試行 実行Runbook（2026-04-02）

このドキュメントは、`character_count=16` かつ `generation_count=50` の3条件実験を、
`git clone` 後すぐ実行できるように手順をまとめたものです。

## 追加済み設定ファイル

- baseline
  - `data/experiment/team_strict_spectrum/report16_50_baseline/data.toml`
  - `data/run_config/experiment_team_strict_spectrum_report16_50_baseline.toml`
- power_wide
  - `data/experiment/team_strict_spectrum/report16_50_power_wide/data.toml`
  - `data/run_config/experiment_team_strict_spectrum_report16_50_power_wide.toml`
- vector_wide
  - `data/experiment/team_strict_spectrum/report16_50_vector_wide/data.toml`
  - `data/run_config/experiment_team_strict_spectrum_report16_50_vector_wide.toml`

## 実行方法（推奨）

### 1) 依存関係

```bash
uv sync
```

### 2) 3条件を連続実行

```bash
uv run python scripts/run_team_strict_spectrum_report16_gen50.py
```

### 3) 1条件だけ実行

```bash
uv run python scripts/run_team_strict_spectrum_report16_gen50.py --only baseline
uv run python scripts/run_team_strict_spectrum_report16_gen50.py --only power_wide
uv run python scripts/run_team_strict_spectrum_report16_gen50.py --only vector_wide
```

## 補足

- 以前の `character_count=16, generation_count=3` 実行時の観測では、おおむね **1試行 ≒ 20秒**。
- そのため、`generation_count=50` は **1条件あたり約1000秒（約16〜17分）**、3条件合計で約50分を見込んでください。
- 各条件は別runとして保存されるため、出力は `results/<run_id>/output/team_strict_spectrum_experiment.json` が3つ作られます（1ファイルに追記されません）。
- スクリプトは3条件分の集約JSONも書き出します。
  - `results/team_strict_spectrum_report16_gen50_latest.json`
  - `results/team_strict_spectrum_report16_gen50_<timestamp>.json`
