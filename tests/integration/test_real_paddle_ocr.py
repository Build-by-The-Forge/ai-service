"""Real PaddleOCR execution on a valid image (Phase 2 Step 1).

The engine reads text rendered into a PNG and must return it. Runs only where
paddleocr and its English PP-OCRv3 models are already present (it never
triggers a download); otherwise it skips and says why.
"""

import importlib.util
from pathlib import Path

import pytest

PADDLE_MODELS = Path.home() / ".paddleocr" / "whl"
LINES = ["Photosynthesis converts light energy", "Question 3. What is osmosis?"]


def _skip_reason() -> str | None:
    if importlib.util.find_spec("paddleocr") is None:
        return "paddleocr is not installed"
    for part in ("det/en", "rec/en"):
        if not any((PADDLE_MODELS / part).glob("*PP-OCRv3*")):
            return f"PaddleOCR {part} PP-OCRv3 model is not present in {PADDLE_MODELS} (no download in tests)"
    return None


pytestmark = pytest.mark.skipif(_skip_reason() is not None, reason=_skip_reason() or "")


def _font():
    from PIL import ImageFont

    for name in ("arial.ttf", "DejaVuSans.ttf", "LiberationSans-Regular.ttf"):
        try:
            return ImageFont.truetype(name, 34)
        except OSError:
            continue
    return ImageFont.load_default(size=34)


@pytest.fixture(scope="module")
def page(tmp_path_factory) -> Path:
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (900, 160), "white")
    draw = ImageDraw.Draw(image)
    for i, line in enumerate(LINES):
        draw.text((20, 20 + i * 62), line, fill="black", font=_font())
    path = tmp_path_factory.mktemp("ocr") / "page.png"
    image.save(path)
    return path


async def test_real_paddleocr_reads_a_rendered_page(page):
    from app.services.ocr.paddle import PaddleOCRService

    lines = await PaddleOCRService().extract_text_from_image(str(page))

    texts = [line["text"].strip().lower() for line in lines]
    assert texts == [line.lower() for line in LINES], texts
    assert all(line["confidence"] >= 0.8 for line in lines), lines
    assert all(len(line["bbox"]) == 4 for line in lines)


async def test_a_blank_page_yields_no_text_not_an_error(tmp_path):
    from PIL import Image

    from app.services.ocr.paddle import PaddleOCRService

    blank = tmp_path / "blank.png"
    Image.new("RGB", (400, 200), "white").save(blank)
    assert await PaddleOCRService().extract_text_from_image(str(blank)) == []
