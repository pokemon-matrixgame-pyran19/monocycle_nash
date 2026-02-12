"""
理論予測テストケースのビルダー

各利得行列について理論的な正しい値を保持し、テストで使用する。
新しいテストケースを追加する場合は、TheoryTestBuilderにメソッドを追加し、
get_all_cases()に登録するだけで、全テストパターンで自動的に検証される。
"""

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class TheoryTestCase:
    """
    理論予測テストケース - この利得行列の理論的な正しい値を保持
    
    Attributes:
        name: テストケース名（例: "janken"）
        powers: パワー値 [p1, p2, ...]
        vectors: 相性ベクトル [(vx1, vy1), ...]
        matrix: 利得行列（理論値）
        equilibrium: 混合戦略の確率分布（ナッシュ均衡解）
        isopower_a: 等パワー座標a (x, y)
        description: 説明文
    """
    name: str
    powers: list[float]
    vectors: list[tuple[float, float]]
    matrix: np.ndarray
    equilibrium: np.ndarray | None
    isopower_a: tuple[float, float] | None
    description: str


class TheoryTestBuilder:
    """
    理論予測テストケースのビルダー
    
    各テストケースの理論値データを定義し、get_all_cases()で一括取得可能。
    新しいケースを追加する際は、新しい@staticmethodを作成し、
    get_all_cases()のリストに追加するだけ。
    """
    
    @staticmethod
    def janken() -> TheoryTestCase:
        """
        じゃんけん: 正三角形配置の等パワー3点
        
        3キャラクターが正三角形を形成し、全てパワー0で等パワー。
        均衡解は各1/3の完全混合戦略。
        """
        ROOT3 = 1.7320508075688772
        
        # 理論値データ
        powers = [0.0, 0.0, 0.0]
        vectors = [(2.0, 0.0), (-1.0, ROOT3), (-1.0, -ROOT3)]
        matrix = np.array([
            [0.0, 2 * ROOT3, -2 * ROOT3],
            [-2 * ROOT3, 0.0, 2 * ROOT3],
            [2 * ROOT3, -2 * ROOT3, 0.0]
        ])
        equilibrium = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])
        isopower_a = (0.0, 0.0)
        
        return TheoryTestCase(
            name="janken",
            powers=powers,
            vectors=vectors,
            matrix=matrix,
            equilibrium=equilibrium,
            isopower_a=isopower_a,
            description="正三角形配置の等パワー3点。均衡解は1/3ずつ。"
        )
    
    @staticmethod
    def get_all_cases() -> list[TheoryTestCase]:
        """
        全テストケースを取得
        
        新しいテストケースを追加する場合は、ここに追加するだけで
        全テストパターンで自動的に検証される。
        
        Returns:
            全TheoryTestCaseのリスト
        """
        return [
            TheoryTestBuilder.janken(),
            # 新しいケースをここに追加
            # TheoryTestBuilder.janken_with_power_shift(),
            # TheoryTestBuilder.four_characters(),
        ]
