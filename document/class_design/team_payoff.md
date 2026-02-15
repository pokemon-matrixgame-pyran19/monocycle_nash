## チーム利得行列の計算（追加設計）

### 概要

チーム利得行列の作成において、通常はキャラクター対戦のゲームを解いて値を設定するが、**二人チーム（2×2利得行列）**の場合は公式を適用できる。さらに**単相性モデルの仮定**を置くと、キャラクター利得行列を経由せず理論式で直接ゲーム値を計算できる。

ここでの 2×2 公式計算・単相性理論式は、適用条件が満たされる場合の**厳密計算**であり、近似ではない。

### team/matrix_approx.py

```python
class TwoPlayerTeamMatrixCalculator:
    """
    二人チーム利得行列の計算
    - 2×2利得行列ではゲーム値公式を適用
    - 単相性モデル仮定では理論式を適用
    """
    - __init__(character_matrix: PayoffMatrix, use_monocycle_formula: bool = True)
    - calculate_team_value(team1: Team, team2: Team): float
    - generate_matrix(teams: list[Team]): PayoffMatrix

class TwoByTwoGameValueCalculator:
    """
    2×2利得行列のゲーム値計算
    - 公式: g = (ad - bc) / (a + d - b - c)
    """
    - calculate(matrix: np.ndarray) -> float
    - calculate_saddle_point(matrix: np.ndarray) -> tuple[float, float] | None

class MonocycleTwoByTwoValueCalculator:
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

class TeamPayoffCalculatorSelector:
    """
    チーム利得行列生成時の計算方法選択
    - Strategy Patternで厳密解/公式解を自動選択
    """
    - __init__(use_monocycle_formula: bool = True)
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

class TwoByTwoFormulaCalculator(TeamPayoffCalculator):
    """2×2ゲーム値公式による計算"""
    - calculate(team1: Team, team2: Team, 
                char_matrix: PayoffMatrix): float

class MonocycleFormulaCalculator(TeamPayoffCalculator):
    """単相性モデル仮定による理論式計算"""
    - calculate(team1: Team, team2: Team, 
                char_matrix: MonocyclePayoffMatrix): float
```

### 利得行列計算のクラス階層

```
TeamPayoffCalculator (抽象基底)
    ├── ExactTeamPayoffCalculator          # 厳密解（全対戦解く）
    ├── TwoByTwoFormulaCalculator          # 2×2公式
    │   └── TwoByTwoGameValueCalculator    # g = (ad-bc)/(a+d-b-c)
    └── MonocycleFormulaCalculator         # 単相性モデル仮定の理論式
        └── MonocycleTwoByTwoValueCalculator  # g = (ef+M)/((v1-v2)×(v3-v4))

TwoPlayerTeamMatrixCalculator
    └── TeamPayoffCalculatorSelector       # 計算方法の自動選択
        ├── ExactTeamPayoffCalculator      # フォールバック用
        ├── TwoByTwoFormulaCalculator      # 一般2×2行列用
        └── MonocycleFormulaCalculator     # 単相性モデル用
```
