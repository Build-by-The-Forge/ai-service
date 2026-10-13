"""OCR input is validated before the engine decodes it (Phase 2 OCR-stack
security step): malformed, unsupported or oversized images fail with
OCRExtractionError instead of a native decoder error or a huge allocation."""

import io
import os
import struct
import zlib

import pytest

from app.core.documents.errors import OCRExtractionError
from app.services.ocr.paddle import PaddleOCRService, validate_image

Image = pytest.importorskip("PIL.Image")


def encoded(fmt: str, size=(64, 32)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, "white").save(buffer, fmt)
    return buffer.getvalue()


def png_chunk(kind: bytes, data: bytes) -> bytes:
    body = kind + data
    return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))


def png_header_only(width: int, height: int) -> bytes:
    """A PNG whose header claims width x height, with no pixel data."""
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + png_chunk(b"IHDR", ihdr)
        + png_chunk(b"IDAT", zlib.compress(b""))
        + png_chunk(b"IEND", b"")
    )


def write(tmp_path, name: str, data: bytes) -> str:
    path = tmp_path / name
    path.write_bytes(data)
    return str(path)


class RecordingEngine:
    def __init__(self):
        self.calls = []

    def ocr(self, image_path, cls):
        self.calls.append(image_path)
        return [[[[[0, 0], [1, 0], [1, 1], [0, 1]], ("text", 0.9)]]]


def service(engine) -> PaddleOCRService:
    instance = object.__new__(PaddleOCRService)  # no paddleocr or models needed
    instance.model = engine
    return instance


@pytest.mark.parametrize("fmt", ["PNG", "JPEG", "TIFF", "WEBP"])
def test_allowed_formats_are_accepted(tmp_path, fmt):
    validate_image(write(tmp_path, f"page.{fmt.lower()}", encoded(fmt)))


def test_content_decides_the_format_not_the_file_name(tmp_path):
    validate_image(write(tmp_path, "really-a-jpeg.png", encoded("JPEG")))


@pytest.mark.parametrize(
    "name, data",
    [
        ("zero-bytes.png", b""),
        ("random.png", os.urandom(4096)),
        ("text.png", b"not an image at all\n"),
    ],
)
def test_unidentifiable_input_is_rejected(tmp_path, name, data):
    with pytest.raises(OCRExtractionError, match="not a valid") as raised:
        validate_image(write(tmp_path, name, data))
    assert raised.value.__cause__ is not None


@pytest.mark.parametrize("fmt", ["PNG", "JPEG"])
def test_truncated_image_is_rejected(tmp_path, fmt):
    data = encoded(fmt, (400, 300))
    with pytest.raises(OCRExtractionError, match="not a valid"):
        validate_image(write(tmp_path, f"truncated.{fmt.lower()}", data[: len(data) // 2]))


@pytest.mark.parametrize("fmt", ["GIF", "BMP"])
def test_formats_outside_the_allow_list_are_rejected(tmp_path, fmt):
    with pytest.raises(OCRExtractionError, match="not a valid"):
        validate_image(write(tmp_path, f"page.{fmt.lower()}", encoded(fmt)))


@pytest.mark.parametrize(
    "width, height",
    [
        (5000, 5000),  # 25M: over our limit, under Pillow's own bomb limit
        (30000, 30000),  # 900M: Pillow refuses it in open(), reported the same way
    ],
)
def test_oversized_image_is_rejected_from_its_header_without_decoding(tmp_path, width, height):
    # The file holds no pixel data, so only a header check (not a decode)
    # can reject it with the size message.
    path = write(tmp_path, "bomb.png", png_header_only(width, height))
    with pytest.raises(OCRExtractionError, match="over the 16000000 pixel limit"):
        validate_image(path)


def test_pixel_limit_is_inclusive(tmp_path):
    path = write(tmp_path, "page.png", encoded("PNG", (100, 50)))
    validate_image(path, max_pixels=5000)
    with pytest.raises(OCRExtractionError, match="over the 4999 pixel limit"):
        validate_image(path, max_pixels=4999)


@pytest.mark.asyncio
async def test_invalid_input_never_reaches_the_engine(tmp_path):
    engine = RecordingEngine()
    with pytest.raises(OCRExtractionError):
        await service(engine).extract_text_from_image(write(tmp_path, "x.png", b""))
    assert engine.calls == []


@pytest.mark.asyncio
async def test_valid_input_reaches_the_engine(tmp_path):
    engine = RecordingEngine()
    path = write(tmp_path, "page.png", encoded("PNG"))
    result = await service(engine).extract_text_from_image(path)
    assert engine.calls == [path]
    assert result[0]["text"] == "text"


@pytest.mark.asyncio
async def test_engine_failure_is_typed_and_keeps_its_cause(tmp_path):
    class FailingEngine:
        def ocr(self, image_path, cls):
            raise AttributeError("'NoneType' object has no attribute 'copy'")

    path = write(tmp_path, "page.png", encoded("PNG"))
    with pytest.raises(OCRExtractionError, match="AttributeError") as raised:
        await service(FailingEngine()).extract_text_from_image(path)
    assert isinstance(raised.value.__cause__, AttributeError)
