# ランダムキャラクター生成による構築相性スペクトラム実験レポート（character_count=16, 2026-04-02）

## 目的

`character_count=16`（`team_size=2`）に拡張したときも、`team = "strict"` の構築行列が

- 最大固有値の突出（dominant gapの大きさ）
- 均衡 support size の小ささ（特に support size ≤ 3）

を維持するか確認する。

## 実験条件

3条件（6キャラ版と同じレンジ設計）で比較。

- baseline16: `p∈[-1,1]`, `v∈[-1,1]`, seed=424242
- power_wide16: `p∈[-2,2]`, `v∈[-0.5,0.5]`, seed=424243
- vector_wide16: `p∈[-0.5,0.5]`, `v∈[-2,2]`, seed=424244

共通設定: `character_count=16`, `team_size=2`, `generation_count=3`。

> 16キャラでは1試行あたり計算時間が長いため、まずは時間計測込みの小規模試行で挙動確認を優先。

## 実行コマンド（時間測定あり）

```bash
uv run python - <<'PY'
import json, time
from pathlib import Path
from monocycle_nash.analysis.app.experiment_team_strict_spectrum import run
from monocycle_nash.runtime.infra.loader.main_config import MainConfigLoader

cfgs=[
('baseline16','data/run_config/experiment_team_strict_spectrum_report16_baseline.toml'),
('power_wide16','data/run_config/experiment_team_strict_spectrum_report16_power_wide.toml'),
('vector_wide16','data/run_config/experiment_team_strict_spectrum_report16_vector_wide.toml'),
]
base=Path('results')
known={p.name for p in base.iterdir()} if base.exists() else set()
for name,cfg in cfgs:
    t0=time.perf_counter()
    code=run(MainConfigLoader(cfg))
    elapsed=time.perf_counter()-t0
    if code!=0:
        raise SystemExit(f'failed: {name}')
    now={p.name for p in base.iterdir()}
    rid=sorted(now-known, key=lambda x:int(x))[-1]
    known=now
    print(name, rid, f"{elapsed:.3f}s")
PY
```

出力 run_id:

- baseline16: `results/1`
- power_wide16: `results/2`
- vector_wide16: `results/3`

## 結果サマリ

| 条件 | 実行時間[s] | dominant_gap_mean | ratio2_to_1_mean | support size ヒストグラム | support_size=3率 | support_size≤3率 | corr(gap, support_size) |
|---|---:|---:|---:|---|---:|---:|---:|
| baseline16 | 60.763 | 2.720 | 0.374 | {1: 1, 3: 2} | 0.667 | 1.000 | -0.977 |
| power_wide16 | 59.833 | 10.591 | 0.094 | {1: 3} | 0.000 | 1.000 | n/a |
| vector_wide16 | 61.160 | 2.123 | 0.474 | {3: 2, 5: 1} | 0.667 | 0.667 | -0.217 |

## 観察

- 実行時間は各条件とも約60秒/3試行（≒20秒/試行）。
- baseline16 と power_wide16 は support size ≤ 3 が 100%。
- power_wide16 は support=1 に集中し、最大固有値突出（dominant gap高め）と疎な支持が同時に出る。
- vector_wide16 では support=5 が1件出て、`v` の広がり増加時に support>3 が出る傾向は6キャラ版と同方向。

## 考察

`character_count=16` でも、少なくとも今回の3条件では「`team = "strict"` で support size ≤ 3 になりやすい」傾向は維持された。

一方で試行数が3と少ないため、統計的な結論よりも**挙動確認（スモーク的検証）**の意味合いが強い。時間見積りは取れたため、次段階は計算資源と相談しつつ試行数を増やす。

## 次アクション

1. `generation_count=10` まで拡張（1条件あたり約200秒見込み）
2. まず baseline16 / power_wide16 の2条件を優先して統計を安定化
3. 計算時間短縮が必要なら、ソルバ選択・早期打ち切り条件・並列化を検討

