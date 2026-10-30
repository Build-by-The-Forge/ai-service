"""A recognition abandoned by the extraction deadline (Phase 3 closeout): it
keeps its image, cannot pile up threads, and how it ended is logged."""

import asyncio
import logging
import os
import tempfile
import threading

import pytest

from app.services.ocr import paddle
from app.services.ocr.paddle import PaddleOCRService

RESULT = [[[[[0, 0], [1, 0], [1, 1], [0, 1]], ("words", 0.9)]]]


class GatedModel:
    """Holds inside recognition until released; records what it could read."""

    def __init__(self, error=None):
        self.entered = threading.Event()
        self.release = threading.Event()
        self.readable = []
        self.calls = 0
        self.error = error

    def ocr(self, image_path, cls=True):
        self.calls += 1
        self.entered.set()
        assert self.release.wait(10)
        self.readable.append((image_path, os.path.getsize(image_path) > 0))
        if self.error:
            raise self.error
        return RESULT


@pytest.fixture
def private_root(tmp_path, monkeypatch, caplog):
    caplog.set_level(logging.WARNING)
    root = tmp_path / "private"
    root.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(root))
    monkeypatch.setattr(paddle, "validate_image", lambda path: None)  # not what is tested
    executor = paddle.ThreadPoolExecutor(max_workers=1, thread_name_prefix="ocr-test")
    monkeypatch.setattr(PaddleOCRService, "_executor", executor)
    yield root
    executor.shutdown(wait=False)


def service(model) -> PaddleOCRService:
    instance = PaddleOCRService.__new__(PaddleOCRService)
    instance.model = model
    return instance


def image(tmp_path, name="page.png") -> str:
    path = tmp_path / name
    path.write_bytes(b"image bytes")
    return str(path)


async def wait_until(condition, timeout=5.0):
    for _ in range(int(timeout / 0.01)):
        if condition():
            return
        await asyncio.sleep(0.01)
    pytest.fail("condition not reached")


async def abandon(ocr, model, path):
    running = asyncio.create_task(ocr.extract_text_from_image(path))
    await wait_until(model.entered.is_set)
    running.cancel()
    with pytest.raises(asyncio.CancelledError):
        await running


async def test_a_normal_call_returns_its_result_and_leaves_no_copy(tmp_path, private_root):
    model = GatedModel()
    model.release.set()
    lines = await service(model).extract_text_from_image(image(tmp_path))
    assert lines[0]["text"] == "words"
    await wait_until(lambda: not list(private_root.iterdir()))


async def test_the_engine_sees_the_images_file_name(tmp_path, private_root):
    model = GatedModel()
    model.release.set()
    await service(model).extract_text_from_image(image(tmp_path, "scan.page.png"))
    assert os.path.basename(model.readable[0][0]) == "scan.page.png"


async def test_an_abandoned_call_keeps_its_image_and_its_end_is_logged(
    tmp_path, private_root, caplog
):
    model = GatedModel()
    path = image(tmp_path)
    await abandon(service(model), model, path)
    assert "abandoned while still running" in caplog.text

    os.unlink(path)  # what the extraction does when cancelled
    model.release.set()
    await wait_until(lambda: "Abandoned OCR call finished" in caplog.text)
    assert model.readable[0][1] is True  # it could still read its image
    await wait_until(lambda: not list(private_root.iterdir()))  # and then it is gone


async def test_an_abandoned_calls_failure_is_logged(tmp_path, private_root, caplog):
    model = GatedModel(error=RuntimeError("native failure"))
    await abandon(service(model), model, image(tmp_path))
    model.release.set()
    await wait_until(lambda: "Abandoned OCR call failed" in caplog.text)
    assert "OCRExtractionError" in caplog.text


async def test_repeated_abandonment_cannot_pile_up_threads_or_work(tmp_path, private_root):
    """A hung call holds the one OCR thread; later calls queue, are cancelled
    before they start, leave nothing behind, and the shared pool stays free."""
    model = GatedModel()
    ocr = service(model)
    path = image(tmp_path)
    await abandon(ocr, model, path)
    ocr_threads = [t for t in threading.enumerate() if t.name.startswith("ocr-test")]
    for _ in range(5):
        later = asyncio.create_task(ocr.extract_text_from_image(path))
        await asyncio.sleep(0.02)
        later.cancel()
        with pytest.raises(asyncio.CancelledError):
            await later
    assert [t for t in threading.enumerate() if t.name.startswith("ocr-test")] == ocr_threads
    assert len(ocr_threads) == 1
    assert len(list(private_root.iterdir())) == 1  # only the running call's copy
    assert await asyncio.wait_for(asyncio.to_thread(lambda: "free"), timeout=2) == "free"

    model.release.set()
    await wait_until(lambda: not list(private_root.iterdir()))
    assert model.calls == 1  # the cancelled ones never ran
