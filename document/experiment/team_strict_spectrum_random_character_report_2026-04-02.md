# ランダムキャラクター生成による構築相性スペクトラム実験レポート（2026-04-02）

## 目的

ランダム生成した6キャラクター（各キャラに `p` と2次元 `v`）から `team = "strict"` の構築相性行列（15x15）を作り、次を確認する。

1. 構築相性行列の固有値スペクトラム（主要固有値比、dominant gap）
2. ナッシュ均衡の support size（**support size ≤ 3** を主指標に確認）
3. `team = "strict"` で作る構築行列が「最大固有値突出 & support size ≤ 3 になりやすいか」の確認

## 実験条件

`experiment_team_strict_spectrum` feature を3条件で起動した。

- Baseline: `p∈[-1,1]`, `v∈[-1,1]`, seed=42
- Power wide: `p∈[-2,2]`, `v∈[-0.5,0.5]`, seed=4242
- Vector wide: `p∈[-0.5,0.5]`, `v∈[-2,2]`, seed=7

すべて `character_count=6`, `team_size=2`, `generation_count=50`。

## 実行コマンド（時間測定あり）

```bash
uv run python - <<'PY'
import time
from monocycle_nash.analysis.app.experiment_team_strict_spectrum import run
from monocycle_nash.runtime.infra.loader.main_config import MainConfigLoader
cfgs=[
('baseline','data/run_config/experiment_team_strict_spectrum_report_baseline.toml'),
('power_wide','data/run_config/experiment_team_strict_spectrum_report_power_wide.toml'),
('vector_wide','data/run_config/experiment_team_strict_spectrum_report_vector_wide.toml'),
]
for name,cfg in cfgs:
    t0 = time.perf_counter()
    code = run(MainConfigLoader(cfg))
    elapsed = time.perf_counter() - t0
    if code != 0:
        raise SystemExit(f'failed: {name}')
    print(name, f"{elapsed:.3f}s")
PY
```

出力 run_id:

- baseline: `results/1`
- power_wide: `results/2`
- vector_wide: `results/3`

## 結果サマリ

| 条件 | 実行時間[s] | dominant_gap_mean | ratio2_to_1_mean | support size ヒストグラム | support_size=3率 | support_size≤3率 | corr(gap, support_size) |
|---|---:|---:|---:|---|---:|---:|---:|
| baseline | 21.062 | 5.243 | 0.295 | {1: 38, 3: 12} | 0.24 | 1.00 | -0.365 |
| power_wide | 20.874 | 58.835 | 0.044 | {1: 50} | 0.00 | 1.00 | n/a |
| vector_wide | 21.476 | 7.353 | 0.293 | {1: 18, 3: 29, 5: 3} | 0.58 | 0.94 | -0.520 |

## 観察

- 3条件とも実行時間は約21秒/50試行（≒0.42秒/試行）で、時間制約下でも50試行までは回せる。
- **Power wide 条件**では、dominant gap が極端に大きく（平均58.84）、均衡 support はすべて1点支持（support≤3率=100%）。
- **Baseline 条件**も support は {1,3} のみで、support≤3率=100%。
- **Vector wide 条件**でも support≤3率は94%で、support=5 は3/50件に留まる。

## 仮説に対する考察

本実験の主眼（`team = "strict"` の構築行列では「最大固有値突出 & support size≤3 になりやすいか」）に対しては、**肯定的な結果**だった。

- 3条件とも support≤3 率が非常に高い（100%, 100%, 94%）。
- 特に最大固有値が強く突出した power_wide 条件は、support=1 に完全収束した。
- 条件内相関は負側（baseline -0.365, vector_wide -0.520）で、「突出すると support=3 が増える」というより、**突出が強いほど support が小さくなる（より疎になる）**挙動が示唆された。

従って今回の設定範囲では、

- `team = "strict"` で生成される構築行列は support size≤3 を取りやすい。
- `v` の振れ幅を大きくすると（vector_wide）まれに support>3 が出るが、依然として少数。

## 制約と次アクション

今回は試行数を 10 → 50 へ増やしたが、統計的に十分とは言い切れない。次を推奨する。

1. 各条件を `generation_count=100~500` へ拡大（想定時間: 約0.42秒/試行）
2. seed を複数（例: 10本）回して平均化
3. `support_size<=3` を目的変数にしたロジスティック回帰（説明変数: dominant_gap, ratio2_to_1, ratio3_to_1, p/v の分散）

