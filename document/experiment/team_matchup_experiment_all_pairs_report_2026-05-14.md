# チーム利得行列検証 実験結果（全36組み合わせ）

## 実行設定
- 実行日時: 2026-05-14
- 入力設定: `/home/runner/work/monocycle_nash/monocycle_nash/data/team_matchup_experiment_all_pairs_inline.toml`
- 出力CSV: `/home/runner/work/monocycle_nash/monocycle_nash/result/1/team_matchup_experiment_all_pairs/team_matchup_experiment_csv/team_matchup_experiment_all_pairs.csv`
- 固定チーム i: `team_i = (c1, c2)`
- 可変チーム j: `g1~g9` から2体選択（9C2=36）

## 該当数確認
- 期待行数: 36
- 実測行数: 36
- `j3`,`j4` は出力側で反時計回り順に並べ替えて記録されるため、設定ファイルの member_ids 順と一致しない場合がある。

## 集計結果
- Bij > 0: 9
- Bij < 0: 11
- Bij = 0: 16
- 平均 Bij: -0.222

- CCW=True 該当数: 14 / 平均Bij: 2.048 / 最小: 0.0 / 最大: 4.0
- CCW=False 該当数: 22 / 平均Bij: -1.667 / 最小: -4.0 / 最大: 0.0

## Bij 上位5件
| j_team | j3 | j4 | vj3-vj4 | angle(deg) | CCW | Bij |
|---|---|---|---:|---:|---|---:|
| team_j03 | g4 | g1 | (0.0, 1.0) | 135.0 | True | 4.0 |
| team_j06 | g7 | g1 | (0.0, 2.0) | 135.0 | True | 4.0 |
| team_j24 | g7 | g4 | (0.0, 1.0) | 135.0 | True | 4.0 |
| team_j34 | g8 | g7 | (1.0, 0.0) | 45.0 | True | 4.0 |
| team_j35 | g9 | g7 | (2.0, 0.0) | 45.0 | True | 4.0 |

## Bij 下位5件
| j_team | j3 | j4 | vj3-vj4 | angle(deg) | CCW | Bij |
|---|---|---|---:|---:|---|---:|
| team_j17 | g3 | g5 | (1.0, -1.0) | 0.0 | False | -4.0 |
| team_j18 | g3 | g6 | (0.0, -1.0) | -45.0 | False | -4.0 |
| team_j19 | g3 | g7 | (2.0, -2.0) | 0.0 | False | -4.0 |
| team_j20 | g3 | g8 | (1.0, -2.0) | -18.4 | False | -4.0 |
| team_j21 | g3 | g9 | (0.0, -2.0) | -45.0 | False | -4.0 |

## 所見（簡潔）
- ご指摘どおり、jは `g1~g9` の2組合せで36件になり、今回の出力も36行を確認。
- CCW=True 側は平均Bijが正で、CCW=False 側は平均Bijが負寄りとなり、向きと有利不利の関連が確認できる。
