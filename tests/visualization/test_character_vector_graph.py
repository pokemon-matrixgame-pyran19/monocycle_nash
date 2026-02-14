from monocycle_nash.character import Character, MatchupVector
from monocycle_nash.visualization import CharacterVectorGraphPlotter


def test_draw_creates_svg_with_labels_and_power_scaled_circles(tmp_path) -> None:
    characters = [
        Character(1.0, MatchupVector(-1.0, 0.5), "キャラA"),
        Character(3.0, MatchupVector(0.5, -0.2), "キャラB"),
        Character(9.0, MatchupVector(1.2, 0.8), "キャラC"),
    ]
    plotter = CharacterVectorGraphPlotter(characters)

    output = tmp_path / "character_vectors.svg"
    saved = plotter.draw(output)

    assert saved.exists()
    content = saved.read_text(encoding="utf-8")
    assert "キャラA" in content
    assert "キャラB" in content
    assert "キャラC" in content

    # powerの大小に応じて半径が増えることを確認
    assert 'r="18.00"' in content
    assert 'r="46.00"' in content


def test_plotter_requires_non_empty_characters() -> None:
    try:
        CharacterVectorGraphPlotter([])
        assert False, "ValueError が発生するべき"
    except ValueError as exc:
        assert "1件以上" in str(exc)
