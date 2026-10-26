"""PaddleOCR recognition runs in a worker thread, so the event loop (worker
heartbeats, callbacks) keeps running during OCR (P3-2, audit F3/F4)."""

import asyncio
import time

from app.services.ocr import paddle
from app.services.ocr.paddle import PaddleOCRService


class BlockingModel:
    """Stands in for PaddleOCR: blocks the calling thread like real OCR."""

    def ocr(self, image_path, cls=True):
        time.sleep(0.3)
        return [[[[[0, 0], [1, 0], [1, 1], [0, 1]], ("text", 0.9)]]]


async def test_the_event_loop_keeps_running_during_recognition(tmp_path, monkeypatch):
    image = tmp_path / "page.png"
    image.write_bytes(b"not decoded: validation is stubbed")
    monkeypatch.setattr(paddle, "validate_image", lambda path: None)
    service = object.__new__(PaddleOCRService)  # no paddleocr or models needed
    service.model = BlockingModel()
    ticks = 0

    async def heartbeat():
        nonlocal ticks
        while True:
            ticks += 1
            await asyncio.sleep(0.02)

    beating = asyncio.create_task(heartbeat())
    lines = await service.extract_text_from_image(str(image))
    beating.cancel()

    assert lines == [{"text": "text", "confidence": 0.9, "bbox": [[0, 0], [1, 0], [1, 1], [0, 1]]}]
    assert ticks >= 5  # the loop ran throughout the 0.3 s of OCR
