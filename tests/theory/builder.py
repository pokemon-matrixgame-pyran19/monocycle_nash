"""
理論予測テストケースのビルダー

各利得行列について理論的な正しい値を保持し、テストで使用する。
新しいテストケースを追加する場合は、TheoryTestBuilderにメソッドを追加し、
get_all_cases()に登録するだけで、全テストパターンで自動的に検証される。
"""

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class TestVariant:
    """
    同じ利得行列の異なる表現（パワー・ベクトルの組）

    利得行列は同じでも、パワーとベクトルの組み合わせは複数ありうる。
    このクラスはその一つの表現を保持する。

    Attributes:
        name: variant名（例: "original", "shifted"）
        powers: パワー値 [p1, p2, ...]
        vectors: 相性ベクトル [(vx1, vy1), ...]
        isopower_a: このvariantから等パワーへの変換ベクトル (x, y)
    """

    name: str
    powers: list[float]
    vectors: list[tuple[float, float]]
    isopower_a: tuple[float, float] | None = None


@dataclass(frozen=True)
class TheoryTestCase:
    """
    理論予測テストケース - 複数variantを持つ

    同じ利得行列に対する複数のパワー・ベクトル表現を保持し、
    それぞれでテストを実行できる。

    Attributes:
        name: テストケース名（例: "janken"）
        variants: 同じ行列の異なる表現（TestVariantのリスト）
        matrix: 利得行列（理論値）- 全variantで共通
        equilibrium: 混合戦略の確率分布（ナッシュ均衡解）
        description: 説明文
        transformations: variant間の変換関係 [(from_idx, to_idx, shift_vector), ...]
    """

    name: str
    variants: list[TestVariant]
    matrix: np.ndarray
    equilibrium: np.ndarray | None
    description: str
    transformations: list[tuple[int, int, tuple[float, float]]] | None = None

    @property
    def primary_variant(self) -> TestVariant:
        """主variant（最初のvariant）を取得"""
        return self.variants[0]

    @property
    def powers(self) -> list[float]:
        """後方互換性: 主variantのpowersを返す"""
        return self.primary_variant.powers

    @property
    def vectors(self) -> list[tuple[float, float]]:
        """後方互換性: 主variantのvectorsを返す"""
        return self.primary_variant.vectors

    @property
    def isopower_a(self) -> tuple[float, float] | None:
        """後方互換性: 主variantのisopower_aを返す"""
        return self.primary_variant.isopower_a


@dataclass(frozen=True)
class TeamTheoryTestCase:
    """構築(Team)利得行列の理論予測テストケース。"""

    name: str
    character_case: TheoryTestCase
    team_labels: list[str]
    team_members: list[tuple[int, int]]
    matrix: np.ndarray
    equilibrium: np.ndarray | None
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

        2つのvariantを持つ:
        - original: [(3.0, 1.0), (0.0, 1.+ROOT3), (0.0, 1.-ROOT3)]
        - shifted:  [(2.0, 0.0), (-1.0, ROOT3), (-1.0, -ROOT3)]

        originalを(-1, -1)だけ平行移動するとshiftedになる。
        """
        root3 = 1.7320508075688772

        matrix = np.array(
            [
                [0.0, 2 * root3, -2 * root3],
                [-2 * root3, 0.0, 2 * root3],
                [2 * root3, -2 * root3, 0.0],
            ]
        )
        equilibrium = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])

        variant_original = TestVariant(
            name="original",
            powers=[-2.0, 1.0 + root3, 1.0 - root3],
            vectors=[(3.0, 1.0), (0.0, 1.0 + root3), (0.0, 1.0 - root3)],
            isopower_a=(1.0, 1.0),
        )

        variant_shifted = TestVariant(
            name="shifted",
            powers=[0.0, 0.0, 0.0],
            vectors=[(2.0, 0.0), (-1.0, root3), (-1.0, -root3)],
            isopower_a=(0.0, 0.0),
        )

        transformations = [(0, 1, (1.0, 1.0))]

        return TheoryTestCase(
            name="janken",
            variants=[variant_original, variant_shifted],
            matrix=matrix,
            equilibrium=equilibrium,
            description="正三角形配置の等パワー3点。2つのvariantで同一行列を表現。",
            transformations=transformations,
        )

    @staticmethod
    def extended_janken() -> TheoryTestCase:
        """
        拡張じゃんけん: 弱いキャラクター（p4）を追加した4点

        じゃんけんの利得行列A'に対して弱いキャラクターp4を追加した行列A。
        p4は他の全てのキャラクターに対して不利な位置に配置される。
        """
        root3 = 1.7320508075688772

        base = np.array(
            [
                [0.0, 1.0, -1.0, 0.0],
                [-1.0, 0.0, 1.0, 0.0],
                [1.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        )
        offset = np.array(
            [
                [0.0, 0.0, 0.0, 2.1],
                [0.0, 0.0, 0.0, -0.9],
                [0.0, 0.0, 0.0, -0.9],
                [-2.1, 0.9, 0.9, 0.0],
            ]
        )
        matrix = 2 * root3 * base + offset

        equilibrium = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 0.0])

        variant1 = TestVariant(
            name="document",
            powers=[0.0, 0.0, 0.0, -0.1],
            vectors=[(2.0, 0.0), (-1.0, root3), (-1.0, -root3), (0.0, 1.0)],
            isopower_a=(0.0, 0.0),
        )

        variant2 = TestVariant(
            name="alternative",
            powers=[-2.0, 1.0 + root3, 1.0 - root3, 0.9],
            vectors=[(3.0, 1.0), (0.0, 1.0 + root3), (0.0, 1.0 - root3), (1.0, 2.0)],
            isopower_a=(1.0, 1.0),
        )

        return TheoryTestCase(
            name="extended_janken",
            variants=[variant1, variant2],
            matrix=matrix,
            equilibrium=equilibrium,
            description="じゃんけんに弱いキャラクターp4を追加した4点。均衡解ではp4は使用されない。",
            transformations=None,
        )

    @staticmethod
    def team_janken() -> TeamTheoryTestCase:
        """じゃんけんの構築ゲーム理論値。戦略順は 12,23,13。"""
        character_case = TheoryTestBuilder.janken()
        root3 = 1.7320508075688772
        matrix = 2 * root3 * np.array(
            [
                [0.0, 1.0, -1.0],
                [-1.0, 0.0, 1.0],
                [1.0, -1.0, 0.0],
            ]
        )
        return TeamTheoryTestCase(
            name="team_janken",
            character_case=character_case,
            team_labels=["12", "23", "13"],
            team_members=[(0, 1), (1, 2), (0, 2)],
            matrix=matrix,
            equilibrium=np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]),
            description="じゃんけん3キャラから作る2人構築ゲーム。",
        )

    @staticmethod
    def team_extended_janken() -> TeamTheoryTestCase:
        """拡張じゃんけんの構築ゲーム理論値。戦略順は 12,13,14,23,24,34。"""
        character_case = TheoryTestBuilder.extended_janken()
        root3 = 1.7320508075688772
        matrix = np.array(
            [
                [0.0, 2 * root3, -0.9, -2 * root3, 2.1, 1.2],
                [-2 * root3, 0.0, -0.9, 2 * root3, 1.2, 2.1],
                [0.9, 0.9, 0.0, 1.8, 2 * root3, -2 * root3],
                [2 * root3, -2 * root3, -1.8, 0.0, -0.9, -0.9],
                [-2.1, -1.2, -2 * root3, 0.9, 0.0, 2 * root3],
                [-1.2, -2.1, 2 * root3, 0.9, -2 * root3, 0.0],
            ]
        )
        return TeamTheoryTestCase(
            name="team_extended_janken",
            character_case=character_case,
            team_labels=["12", "13", "14", "23", "24", "34"],
            team_members=[(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)],
            matrix=matrix,
            equilibrium=np.array([1.0 / 3.0, 1.0 / 3.0, 0.0, 1.0 / 3.0, 0.0, 0.0]),
            description="拡張じゃんけん4キャラから作る2人構築ゲーム。",
        )

    @staticmethod
    def get_all_cases() -> list[TheoryTestCase]:
        """全キャラクター行列テストケースを取得。"""
        return [
            TheoryTestBuilder.janken(),
            TheoryTestBuilder.extended_janken(),
        ]

    @staticmethod
    def get_all_team_cases() -> list[TeamTheoryTestCase]:
        """全構築(Team)行列テストケースを取得。"""
        return [
            TheoryTestBuilder.team_janken(),
            TheoryTestBuilder.team_extended_janken(),
        ]
