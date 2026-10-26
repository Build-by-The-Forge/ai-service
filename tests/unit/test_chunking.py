"""Normalization keeps every symbol; chunks map to their pages and have
deterministic ids (P3-3)."""

import pytest

from app.services.ingestion.chunking import chunk_document, chunk_id
from app.services.ingestion.extraction import ExtractedPage
from app.services.ingestion.normalize import normalize_text


def page(number: int, text: str) -> ExtractedPage:
    return ExtractedPage(number=number, kind="native", method="text_layer", text=text)


def chunks(pages, **kwargs):
    return chunk_document(pages, tenant_id="t-1", asset_id="a-1", run_number=1, **kwargs)


# --- normalization -------------------------------------------------------------------------


def test_mathematical_symbols_and_greek_survive() -> None:
    text = "x² + y² ≤ r²,  ∫₀¹ f(x) dx = π/4 ± ε;  α β γ Δ θ λ µ σ Ω  ½ ∞ √2 ≠ ≈ → ∈"
    assert (
        normalize_text(text)
        == "x² + y² ≤ r², ∫₀¹ f(x) dx = π/4 ± ε; α β γ Δ θ λ µ σ Ω ½ ∞ √2 ≠ ≈ → ∈"
    )


def test_whitespace_and_control_characters_are_tidied() -> None:
    assert normalize_text("a\t\tb   c \r\n\r\n\r\n\r\nd\x00e\x07") == "a b c\n\nde"


def test_a_word_broken_at_a_line_end_is_joined_but_a_minus_is_not() -> None:
    assert normalize_text("differen-\ntiation") == "differentiation"
    assert normalize_text("x -\n1") == "x -\n1"
    assert normalize_text("Euler-\nLagrange") == "Euler-\nLagrange"


def test_composed_and_decomposed_text_normalize_alike() -> None:
    assert normalize_text("café") == normalize_text("café") == "café"


# --- chunking ------------------------------------------------------------------------------


def test_short_pages_pack_into_one_chunk_with_their_page_span() -> None:
    result = chunks([page(1, "Limits of functions."), page(2, "Continuity on intervals.")])

    assert len(result) == 1
    assert (result[0].page_start, result[0].page_end) == (1, 2)
    assert result[0].text == "Limits of functions.\n\nContinuity on intervals."


def test_every_chunk_maps_back_to_its_pages_and_respects_the_size() -> None:
    pages = [
        page(n, "\n\n".join(f"Page {n} paragraph {i}. " * 6 for i in range(8))) for n in range(1, 6)
    ]

    result = chunks(pages, max_chars=500, overlap_chars=100)

    assert len(result) > 5
    for chunk in result:
        assert len(chunk.text) <= 500
        pages_named = {
            int(word) for word in chunk.text.replace(".", " ").split()[1::4] if word.isdigit()
        }
        assert chunk.page_start <= chunk.page_end
        assert pages_named <= set(range(chunk.page_start, chunk.page_end + 1))
    assert result[0].page_start == 1 and result[-1].page_end == 5


def test_a_paragraph_longer_than_a_chunk_is_split_at_sentences() -> None:
    sentences = [f"Sentence number {i} explains a step of the proof." for i in range(40)]
    result = chunks([page(1, " ".join(sentences))], max_chars=300, overlap_chars=0)

    assert all(len(c.text) <= 300 for c in result)
    assert all(c.text.endswith(".") for c in result)
    assert " ".join(c.text for c in result) == " ".join(sentences)  # nothing lost, no overlap


def test_a_single_unbroken_line_is_split_at_whitespace() -> None:
    text = "word " * 400
    result = chunks([page(1, text)], max_chars=200, overlap_chars=0)

    assert all(len(c.text) <= 200 for c in result)
    assert sum(c.text.count("word") for c in result) == 400


def test_consecutive_chunks_overlap_by_whole_sentences() -> None:
    sentences = [f"Fact {i} about electric fields is stated here." for i in range(30)]
    result = chunks([page(1, "\n\n".join(sentences))], max_chars=400, overlap_chars=120)

    for before, after in zip(result, result[1:]):
        last_sentence = before.text.split("\n\n")[-1]
        assert after.text.startswith(last_sentence)


def test_the_same_input_gives_the_same_chunks_and_ids() -> None:
    pages = [page(1, "Alpha. " * 300), page(2, "Beta. " * 300)]

    first, second = chunks(pages, max_chars=400), chunks(pages, max_chars=400)

    assert first == second
    assert [c.chunk_id for c in first] == [chunk_id("t-1", "a-1", 1, i) for i in range(len(first))]


def test_ids_differ_by_tenant_asset_and_run() -> None:
    ids = {chunk_id(t, a, r, 0) for t in ("t-1", "t-2") for a in ("a-1", "a-2") for r in (1, 2)}
    assert len(ids) == 8


def test_empty_pages_make_no_chunks() -> None:
    assert chunks([page(1, ""), page(2, "   \n\n  ")]) == []


def test_the_overlap_must_be_smaller_than_a_chunk() -> None:
    with pytest.raises(ValueError):
        chunks([page(1, "x")], max_chars=100, overlap_chars=100)
