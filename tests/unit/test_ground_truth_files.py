"""The OCR ground-truth files are internally consistent (PR #17 review).

Every page has a transcription, every text region is a verbatim excerpt of
its page's full_text (so region metrics cannot drift from the transcription),
no template placeholder remains, and each file records its review and,
once verified, who verified it, when and against which revision.
This checks consistency only; whether a transcription matches its page is a
human verification, recorded in each file's metadata.
"""

import json
from pathlib import Path

import pytest

GROUND_TRUTH = Path(__file__).resolve().parents[2] / "data" / "ocr_evaluation" / "ground_truth"
FILES = sorted(GROUND_TRUTH.glob("*.json"))
PLACEHOLDERS = (
    "[Question",
    "[Text content",
    "[Complete worked",
    "[Section heading",
    "[Mathematical expression",
    "[Expected OCR",
)


def pages(name: Path) -> list[dict]:
    truth = json.loads(name.read_text(encoding="utf-8"))["ground_truth"]
    return truth.get("pages") or [truth]


def test_the_ground_truth_set_is_present():
    assert len(FILES) == 7
    assert sum(len(pages(f)) for f in FILES) == 8


@pytest.mark.parametrize("path", FILES, ids=[f.stem for f in FILES])
def test_every_region_is_a_verbatim_excerpt_of_its_page(path):
    for page in pages(path):
        text = page["full_text"]
        assert text.strip(), f"page {page.get('page_number', 1)} has no transcription"
        assert not any(marker in text for marker in PLACEHOLDERS)
        for region in page.get("text_regions", []):
            assert region["text"] in text, region["id"]
            assert not region["text"].startswith("["), region["id"]


@pytest.mark.parametrize("path", FILES, ids=[f.stem for f in FILES])
def test_the_verification_is_recorded(path):
    metadata = json.loads(path.read_text(encoding="utf-8"))["metadata"]
    assert metadata.get("review_notes"), "the file must say what was reviewed"
    if metadata["ground_truth_type"] == "draft_pending_human_verification":
        assert metadata["verified_by"] is None
    else:
        assert metadata["ground_truth_type"] == "human_verified"
        verifier = metadata["verified_by"]
        assert {"name", "date", "revision", "basis"} <= verifier.keys()
        assert all(verifier[key] for key in ("name", "date", "revision", "basis"))
