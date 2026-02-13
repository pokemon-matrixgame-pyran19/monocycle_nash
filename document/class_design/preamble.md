# クラス設計ドキュメント v3

## 前提と設計方針

### 利得行列の構造

利得行列の行・列は「戦略」に対応。戦略には以下を設定可能：
- **キャラクター戦略**: 1体のキャラクター
- **チーム戦略**: 複数キャラクターのチーム（Team）

### 利得行列の2種類

**一般の利得行列 (GeneralPayoffMatrix)**
- 任意の行列要素 Aij を持つ
- 解法: nashpyによる線形最適化（汎用的だが計算コスト高）

**単相性モデルの利得行列 (MonocyclePayoffMatrix)**
- MonocycleCharacter (p, v) から生成: Aij = pi - pj + vi×vj
- 解法: 等パワー座標による高速解法（構造的性質を活用）
- 元のCharacter情報を保持

### キャラクターの2種類

**MonocycleCharacter（単相性モデル用）**
- power: パワー値
- vector: 相性ベクトル(vx, vy)
- 等パワー座標計算に使用

**GenericCharacter（一般・表示用）**
- label: 表示用ラベル
- パラメータを持たない軽量なキャラクター

### 設計方針

1. **戦略の抽象化**: 利得行列は Character/Team ではなく PureStrategy を扱う
2. **利得行列を型で区別**: 2種類の利得行列を別クラスとして定義
3. **Solverは行列型に応じて自動選択**: Strategy Pattern
4. **単相性モデル専用の最適化**: MonocyclePureStrategy のパラメータを活かした高速計算

---

## ディレクトリ構造

```
src/
├── monocycle_nash/            # パッケージルート
│   ├── __init__.py
│   ├── character/             # キャラクター関連
│   │   ├── __init__.py
│   │   ├── domain.py          # Character, MatchupVector
│   │   └── calculator.py      # パワー・ベクトル計算
│   ├── matrix/                # 利得行列関連（コア層）
│   │   ├── __init__.py
│   │   ├── base.py            # PayoffMatrix抽象基底クラス
│   │   ├── general.py         # 一般利得行列
│   │   ├── monocycle.py       # 単相性モデル利得行列
│   │   ├── pool.py            # キャラクタープール管理
│   │   └── builder.py         # 利得行列ビルダー
│   ├── solver/                # 均衡解ソルバー（Strategy Pattern）
│   │   ├── __init__.py
│   │   ├── base.py            # EquilibriumSolver抽象クラス
│   │   ├── selector.py        # 行列型に応じたソルバー選択
│   │   ├── nashpy_solver.py   # 一般行列用（nashpy線形最適化）
│   │   └── isopower_solver.py # 単相性モデル用（等パワー座標高速解法）
│   ├── isopower/              # 等パワー座標関連（単相性モデル専用）
│   │   ├── __init__.py
│   │   ├── coordinate.py      # 座標変換
│   │   ├── a_calculator.py    # aベクトル計算
│   │   ├── triangle.py        # 最適三角形探索
│   │   └── evaluator.py       # 評価
│   ├── equilibrium/           # 均衡解の表現
│   │   ├── __init__.py
│   │   ├── domain.py          # MixedStrategy
│   │   └── validator.py       # 均衡解の検証
│   ├── strategy/              # 純粋戦略関連
│   │   ├── __init__.py
│   │   └── domain.py          # PureStrategy, PureStrategySet
│   ├── team/                  # チーム関連
│   │   ├── __init__.py
│   │   ├── domain.py          # Team
│   │   ├── factory.py         # チーム生成ファクトリ
│   │   └── matrix.py          # チーム利得行列
│   └── visualizer/            # 可視化
│       ├── __init__.py
│       ├── character.py
│       ├── matrix.py
│       ├── equilibrium.py
│       ├── isopower.py
│       └── strategy.py
├── main.py                    # アプリケーションエントリーポイント
└── example.py                 # 使用例
```

**インポート例:**
```python
from monocycle_nash.character.domain import Character, MatchupVector
from monocycle_nash.matrix.monocycle import MonocyclePayoffMatrix
from monocycle_nash.solver.selector import SolverSelector
```
