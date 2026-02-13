## チーム利得行列の近似計算（追加設計）

### 概要

チーム利得行列の作成において、通常はキャラクター対戦のゲームを解いて値を設定するが、**二人チーム（2×2利得行列）**の場合は公式を用いて高速に計算可能。さらに**単相性モデルの仮定**を追加すると、より簡潔な計算が可能になる。

### team/matrix_approx.py

```python
class TwoPlayerTeamMatrixApproximator:
    """
    二人チーム利得行列の近似計算
    - 2×2利得行列のゲーム値公式を使用して高速化
    - 単相性モデル仮定でさらに最適化
    """
    - __init__(character_matrix: PayoffMatrix)
    - calculate_team_value(team1: Team, team2: Team): float
    - generate_approx_matrix(teams: list[Team]): PayoffMatrix

class TwoByTwoGameValueCalculator:
    """
    2×2利得行列のゲーム値計算
    - 公式: g = (ad - bc) / (a + d - b - c)
    """
    - calculate(matrix: np.ndarray) -> float
    - calculate_saddle_point(matrix: np.ndarray) -> tuple[float, float] | None

class MonocycleTwoByTwoApproximator:
    """
    単相性モデル仮定下的2×2ゲーム値計算
    - パラメータ: p1, p2, p3, p4, v1, v2, v3, v4
    - 公式: g = (ef + M) / ((v1-v2) × (v3-v4))
      ここで e = p1-p2 + v1×v2, f = p3-p4 + v3×v4
      M は4×4行列式 |1 1 1 1; p1 p2 p3 p4; v1 v2 v3 v4|
    """
    - __init__(c1: MonocycleCharacter, c2: MonocycleCharacter, 
               c3: MonocycleCharacter, c4: MonocycleCharacter)
    - calculate_game_value(): float
    - calculate_e_parameter(): float
    - calculate_f_parameter(): float
    - calculate_m_determinant(): float

class TeamPayoffFormulaSelector:
    """
    チーム利得行列生成時の計算方法選択
    - Strategy Patternで厳密解/近似解を自動選択
    """
    - __init__(use_monocycle_approx: bool = True)
    - select_calculator(team1: Team, team2: Team, 
                        char_matrix: PayoffMatrix) -> TeamPayoffCalculator
    - calculate(team1: Team, team2: Team, char_matrix: PayoffMatrix): float

class TeamPayoffCalculator(ABC):
    """チーム利得計算の抽象基底クラス"""
    - @abstractmethod calculate(team1: Team, team2: Team, 
                                char_matrix: PayoffMatrix): float

class ExactTeamPayoffCalculator(TeamPayoffCalculator):
    """厳密なチーム利得計算（全キャラクター対戦を解く）"""
    - calculate(team1: Team, team2: Team, 
                char_matrix: PayoffMatrix): float

class TwoByTwoApproximateCalculator(TeamPayoffCalculator):
    """2×2ゲーム値公式による近似計算"""
    - calculate(team1: Team, team2: Team, 
                char_matrix: PayoffMatrix): float

class MonocycleApproximateCalculator(TeamPayoffCalculator):
    """単相性モデル仮定による最速近似計算"""
    - calculate(team1: Team, team2: Team, 
                char_matrix: MonocyclePayoffMatrix): float
```

### 利得行列近似計算のクラス階層

```
TeamPayoffCalculator (抽象基底)
    ├── ExactTeamPayoffCalculator          # 厳密解（全対戦解く）
    ├── TwoByTwoApproximateCalculator      # 2×2公式による近似
    │   └── TwoByTwoGameValueCalculator    # g = (ad-bc)/(a+d-b-c)
    └── MonocycleApproximateCalculator     # 単相性モデル仮定による最速近似
        └── MonocycleTwoByTwoApproximator  # g = (ef+M)/((v1-v2)×(v3-v4))

TwoPlayerTeamMatrixApproximator
    └── TeamPayoffFormulaSelector          # 計算方法の自動選択
        ├── ExactTeamPayoffCalculator      # フォールバック用
        ├── TwoByTwoApproximateCalculator  # 一般2×2行列用
        └── MonocycleApproximateCalculator # 単相性モデル用
```
