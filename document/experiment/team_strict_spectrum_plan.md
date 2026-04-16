# team="strict" 構築利得行列のスペクトル実験計画

## 目的
- 6キャラクターから2匹チーム（\(\binom{6}{2}=15\) 戦略）を生成し、`team="strict"` による構築利得行列を多数サンプルする。
- 各行列に対して以下を観測する。
  - 固有値指標（交代行列の純虚固有値の絶対値）
    - \(\lambda_2/\lambda_1\)
    - \(\lambda_3/\lambda_1\)
  - 均衡解の非ゼロ成分数（support size）
- 仮説:
  - 最大固有値 \(\lambda_1\) が突出（\(\lambda_1/\lambda_2\) 大）すると、support size = 3 が増える。

## 設計
- 1試行で以下を実施:
  1. 6キャラの `p` と `v=(x,y)` を一様乱数で生成。
  2. `build_matrix_from_input({"characters": ..., "team": "strict"})` で15x15の構築利得行列を構築。
  3. `SolverSelector().solve(matrix)` で均衡を取得。
  4. `matrix.eigenvalues()` から純虚固有値絶対値を抽出し、数値誤差除去後にユニーク化（共役ペア同一視）。
  5. `\lambda_2/\lambda_1`, `\lambda_3/\lambda_1`, `\lambda_1/\lambda_2`（dominant gap）を計算。
  6. `probabilities > support_threshold` で support size を算出。

## 統計出力
- JSON: `output/team_strict_spectrum_experiment.json`
  - `trials`: 試行ごとの `lambda1`, `lambda2`, `lambda3`, 各比率, support size
  - `summary`: 平均・標準偏差・ヒストグラム・相関・仮説検証用集計
  - `hypothesis_analysis`: `dominant_gap` ビンごとの `support3_rate`
- CSV: `output/team_strict_spectrum_trials.csv`
  - 試行単位の生データ

## 既定パラメータ
- `character_count = 6`
- `team_size = 2`
- `generation_count = 2000`
- `power_low/high = -1.0 / 1.0`
- `vector_low/high = -1.0 / 1.0`
- `support_threshold = 1e-6`
- `random_seed = 42`

## 実行方法
`data/run_config/experiment_team_strict_spectrum.toml` を使って実行する。

```bash
cp data/run_config/experiment_team_strict_spectrum.toml data/run_config/main.toml
uv run main
```


