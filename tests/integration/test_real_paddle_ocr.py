"""Real PaddleOCR execution on a valid image (Phase 2 Steps 1–2).

The engine reads text rendered into a PNG and must return it.

Models: from OCR_MODEL_DIR when set (the verified install the image ships);
then every socket is blocked, so a hidden download fails the test. Without
OCR_MODEL_DIR the paddleocr cache is used if it is already populated. The
test never downloads; it skips and says why, unless OCR_TEST_REQUIRE_MODELS=1
(the offline container gate), which turns a skip into a failure.
"""

import importlib.util
import os
import socket
from pathlib import Path

import pytest

PADDLE_CACHE = Path.home() / ".paddleocr" / "whl"
LINES = ["Photosynthesis converts light energy", "Question 3. What is osmosis?"]
REQUIRE = os.environ.get("OCR_TEST_REQUIRE_MODELS") == "1"


def _missing() -> str | None:
    if importlib.util.find_spec("paddleocr") is None:
        return "paddleocr is not installed"
    model_dir = os.environ.get("OCR_MODEL_DIR")
    if model_dir:
        return (
            None
            if (Path(model_dir) / "installed.json").is_file()
            else f"no installed models in {model_dir}"
        )
    for part in ("det/en", "rec/en"):
        if not any((PADDLE_CACHE / part).glob("*PP-OCRv3*")):
            return f"PaddleOCR {part} PP-OCRv3 model is not present in {PADDLE_CACHE} (no download in tests)"
    return None


MISSING = _missing()
if MISSING and REQUIRE:
    pytest.fail(f"OCR_TEST_REQUIRE_MODELS=1 but {MISSING}", pytrace=False)
pytestmark = pytest.mark.skipif(MISSING is not None, reason=MISSING or "")


LOOPBACK = {"127.0.0.1", "::1", "localhost"}


def _host(address) -> str:
    return str(address[0] if isinstance(address, tuple) else address)


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """With packaged models nothing may reach the network.

    Name resolution and connections to anything but loopback are refused,
    which is what a model download needs. Sockets themselves stay creatable:
    the event loop builds a loopback socket pair for its own use.
    """
    if not os.environ.get("OCR_MODEL_DIR"):
        yield
        return
    import app.services.ocr.paddle as paddle_module

    monkeypatch.setattr(paddle_module.settings, "OCR_MODEL_DIR", os.environ["OCR_MODEL_DIR"])

    real_getaddrinfo = socket.getaddrinfo
    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex

    def refuse(what):
        raise OSError(f"network access attempted during OCR: {what}")

    def getaddrinfo(host, *args, **kwargs):
        if str(host) not in LOOPBACK:
            refuse(f"resolve {host}")
        return real_getaddrinfo(host, *args, **kwargs)

    def connect(self, address):
        if _host(address) not in LOOPBACK:
            refuse(f"connect {address}")
        return real_connect(self, address)

    def connect_ex(self, address):
        if _host(address) not in LOOPBACK:
            refuse(f"connect {address}")
        return real_connect_ex(self, address)

    monkeypatch.setattr(socket, "getaddrinfo", getaddrinfo)
    monkeypatch.setattr(socket.socket, "connect", connect)
    monkeypatch.setattr(socket.socket, "connect_ex", connect_ex)
    yield


def test_the_network_guard_blocks_a_real_download_attempt():
    """The guard itself works: an attempt to fetch a model is refused."""
    if not os.environ.get("OCR_MODEL_DIR"):
        pytest.skip("the network guard is active only with OCR_MODEL_DIR")
    import urllib.request

    with pytest.raises(OSError, match="network access attempted during OCR"):
        urllib.request.urlopen("https://paddleocr.bj.bcebos.com/", timeout=5)


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
