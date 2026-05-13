# 固定構築 vs 全構築 実験レポート（2026-05-13）

## 目的

`i=(c1,c2)` を固定したとき、他の構築 `j` に対する利得 `Bij` の符号と、ベクトル幾何（`v1-v2` と `vj3-vj4` の角度）の関係を確認する。

## 実験条件

- キャラクター数: 11
  - `c1=(x,0)`, `c2=(0,y)`
  - `c3..c11` は原点周りの 9 格子点  
    `(-1,-1),(0,-1),(1,-1),(-1,0),(0,0),(1,0),(-1,1),(0,1),(1,1)`
- 全キャラクターの `power=0`
- 構築は 2 体選出（55通り）。`i=(c1,c2)` を除く 54 通りを `j` として評価
- 計算方法: 2x2 厳密（`TwoPlayerTeamMatrixCalculator(..., use_monocycle_formula=False)`）
- 比較ケース: `(x,y)=(1,1),(2,2),(3,3),(3,1),(1,3)`

## 出力列（ケース別 CSV）

- `j3`, `j4`
- `v_j3`, `v_j4`
- `v_j3_minus_j4`
- `angle_deg_v12_to_v34`（符号付き角度）
- `Bij`

## 集計結果

| x | y | 対戦数 | Bij>0 | Bij<0 | Bij=0 | mean(Bij) | min(Bij) | max(Bij) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 54 | 14 | 15 | 25 | -0.0185 | -1.0 | 1.0 |
| 2 | 2 | 54 | 14 | 15 | 25 | -0.0370 | -2.0 | 2.0 |
| 3 | 3 | 54 | 14 | 15 | 25 | -0.0556 | -3.0 | 3.0 |
| 3 | 1 | 54 | 14 | 15 | 25 | 0.1591 | -1.8 | 3.0 |
| 1 | 3 | 54 | 14 | 15 | 25 | 0.1372 | -1.0 | 3.0 |

## 観察

1. `x=y` を同時に大きくしても、`Bij>0 / Bij<0 / Bij=0` の件数は変化しなかった。  
2. ただし `x=y` を大きくすると `|Bij|`（min/max の絶対値）は増えた。  
3. 非対称ケース `(3,1)`, `(1,3)` では `mean(Bij)` が正に寄った。  
4. 角度符号との関係は、今回の定義ではおおむね  
   - `angle > 0` のとき `Bij > 0` が多い  
   - `angle < 0` のとき `Bij < 0` が多い  
   となり、`「反時計回りなら j 有利（Bij<0）」` という仮説とは逆向きの傾向が強かった。

## 生成物（実行時）

- `/tmp/monocycle_team_experiment/summary.csv`
- `/tmp/monocycle_team_experiment/focus_c1c2_grid9_x1_y1.csv`
- `/tmp/monocycle_team_experiment/focus_c1c2_grid9_x2_y2.csv`
- `/tmp/monocycle_team_experiment/focus_c1c2_grid9_x3_y3.csv`
- `/tmp/monocycle_team_experiment/focus_c1c2_grid9_x3_y1.csv`
- `/tmp/monocycle_team_experiment/focus_c1c2_grid9_x1_y3.csv`
