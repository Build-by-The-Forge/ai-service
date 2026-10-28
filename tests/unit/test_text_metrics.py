import pytest

from app.services.ingestion.text_metrics import cer, edit_distance, normalize, wer


def test_normalization_ignores_layout_but_keeps_every_character() -> None:
    assert normalize("  lim x→2\n\n (x² − 4)\t") == "lim x→2 (x² − 4)"
    assert normalize("é") == "é"  # NFC


@pytest.mark.parametrize(
    ("a", "b", "distance"),
    [("", "", 0), ("abc", "", 3), ("kitten", "sitting", 3), ("x²", "x2", 1), ("π", "π", 0)],
)
def test_edit_distance(a, b, distance) -> None:
    assert edit_distance(a, b) == distance == edit_distance(b, a)


def test_rates_are_relative_to_the_reference() -> None:
    assert cer("abcd", "abcd") == 0.0
    assert cer("abcd", "abXd") == 0.25
    assert wer("the electric field", "the electric feld") == pytest.approx(1 / 3)
    assert cer("abc", "") == 1.0
    assert cer("", "") == 0.0
