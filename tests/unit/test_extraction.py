"""DocumentExtractor on PDFs built here with PyMuPDF and a fake OCR engine
(P3-2): classification, reading order, OCR only where needed, limits and
typed failures."""

import asyncio

import fitz
import pytest

from app.core.documents.errors import OCRDependencyUnavailableError, OCRExtractionError
from app.services.ingestion.extraction import (
    DocumentExtractor,
    classify_page,
    jump_fraction,
    text_layer,
)
from app.services.tasks.ingestion_outcomes import PermanentIngestionError, PipelineUnavailableError

PROSE = "Electric charge is conserved in every interaction we know of. " * 6  # over 200 chars


class FakeOcr:
    def __init__(self, lines=None, error=None, delay=0.0):
        self.lines = lines if lines is not None else [{"text": "scanned words", "confidence": 0.9}]
        self.error = error
        self.delay = delay
        self.calls = 0

    async def extract_text_from_image(self, image_path):
        self.calls += 1
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.error:
            raise self.error
        return self.lines


def extractor(engine=None, *, max_pages=600, timeout=30.0, factory=None):
    engine = engine or FakeOcr()
    return DocumentExtractor(
        ocr_engine_factory=factory or (lambda: engine),
        max_pages=max_pages,
        timeout_seconds=timeout,
        min_confidence=0.5,
    )


def save(doc: fitz.Document, path, **kwargs) -> str:
    doc.save(str(path), **kwargs)
    doc.close()
    return str(path)


def native_pdf(tmp_path, pages=1) -> str:
    doc = fitz.open()
    for _ in range(pages):
        page = doc.new_page()
        page.insert_textbox(fitz.Rect(50, 50, 550, 750), PROSE)
    return save(doc, tmp_path / "native.pdf")


def scanned_pdf(tmp_path) -> str:
    """A page that is only an image of text: no text layer."""
    source = fitz.open()
    source.new_page().insert_textbox(fitz.Rect(50, 50, 550, 750), PROSE)
    picture = source[0].get_pixmap(matrix=fitz.Matrix(1, 1)).tobytes("png")
    source.close()
    doc = fitz.open()
    page = doc.new_page()
    page.insert_image(page.rect, stream=picture)
    return save(doc, tmp_path / "scanned.pdf")


# --- classification and reading order ----------------------------------------------------------


def test_pages_are_classified_by_text_layer_and_image_coverage(tmp_path):
    doc = fitz.open(native_pdf(tmp_path))
    assert classify_page(doc[0]) == "native"
    assert classify_page(fitz.open(scanned_pdf(tmp_path))[0]) == "scanned"
    assert classify_page(fitz.open().new_page()) == "empty"


def test_a_page_drawn_out_of_order_is_read_in_reading_order(tmp_path):
    """Blocks drawn bottom first, as office exports do: sorted by position."""
    doc = fitz.open()
    page = doc.new_page()
    for y, label in ((700, "Third"), (100, "First"), (400, "Second"), (550, "Second-and-a-half")):
        page.insert_text((72, y), f"{label} section " + "x" * 60)
    assert jump_fraction(page) >= 0.05
    text = text_layer(page)
    assert text.index("First") < text.index("Second") < text.index("Third")


def test_a_page_drawn_in_reading_order_keeps_its_order(tmp_path):
    doc = fitz.open()
    page = doc.new_page()
    for y, label in ((100, "Alpha"), (300, "Beta"), (500, "Gamma")):
        page.insert_text((72, y), f"{label} " + "y" * 60)
    assert jump_fraction(page) == 0.0
    text = text_layer(page)
    assert text.index("Alpha") < text.index("Beta") < text.index("Gamma")


def test_ligatures_are_expanded(tmp_path):
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 100), "diﬀerentiation deﬁnition", fontname="helv")
    assert "ﬀ" not in text_layer(page) and "ﬁ" not in text_layer(page)


# --- extraction ------------------------------------------------------------------------------


async def test_a_native_document_never_loads_the_ocr_model(tmp_path):
    def no_ocr():
        raise AssertionError("the OCR engine was created for a native document")

    document = await extractor(factory=no_ocr).extract(
        native_pdf(tmp_path, pages=3), "application/pdf"
    )

    assert [p.method for p in document.pages] == ["text_layer"] * 3
    metrics = document.stage_metrics()
    assert (metrics["method"], metrics["pages"], metrics["native_pages"], metrics["ocr_pages"]) == (
        "native_text",
        3,
        3,
        0,
    )
    assert metrics["characters"] == document.characters > 0


async def test_a_scanned_page_is_ocred(tmp_path):
    engine = FakeOcr(
        [{"text": "first line", "confidence": 0.8}, {"text": "second", "confidence": 0.6}]
    )

    document = await extractor(engine).extract(scanned_pdf(tmp_path), "application/pdf")

    page = document.pages[0]
    assert (page.kind, page.method, page.text, page.confidence) == (
        "scanned",
        "ocr",
        "first line\nsecond",
        0.7,
    )
    assert engine.calls == 1
    assert document.stage_metrics()["method"] == "ocr"


async def test_low_confidence_pages_are_reported_not_dropped(tmp_path):
    engine = FakeOcr([{"text": "faint", "confidence": 0.2}])

    document = await extractor(engine).extract(scanned_pdf(tmp_path), "application/pdf")

    assert document.pages[0].warnings == ("low_confidence",)
    assert document.stage_metrics()["low_confidence_pages"] == 1


async def test_an_image_upload_is_ocred_directly(tmp_path):
    image = tmp_path / "photo.png"
    fitz.open(native_pdf(tmp_path))[0].get_pixmap().save(str(image))

    document = await extractor().extract(str(image), "image/png")

    assert [(p.kind, p.method) for p in document.pages] == [("image", "ocr")]


# --- refusals ----------------------------------------------------------------------------------


async def test_an_encrypted_pdf_is_refused(tmp_path):
    doc = fitz.open()
    doc.new_page().insert_text((72, 100), PROSE)
    path = save(
        doc, tmp_path / "locked.pdf", encryption=fitz.PDF_ENCRYPT_AES_256, user_pw="u", owner_pw="o"
    )

    with pytest.raises(PermanentIngestionError) as caught:
        await extractor().extract(path, "application/pdf")

    assert caught.value.code == "PDF_ENCRYPTED"


async def test_a_pdf_with_only_permission_restrictions_is_read(tmp_path):
    doc = fitz.open()
    doc.new_page().insert_textbox(fitz.Rect(50, 50, 550, 750), PROSE)
    path = save(
        doc,
        tmp_path / "restricted.pdf",
        encryption=fitz.PDF_ENCRYPT_AES_256,
        owner_pw="o",
        permissions=0,
    )

    document = await extractor().extract(path, "application/pdf")

    assert document.pages[0].method == "text_layer"


async def test_a_corrupt_pdf_is_refused(tmp_path):
    path = tmp_path / "broken.pdf"
    path.write_bytes(b"%PDF-1.7\nthis is not a pdf at all")

    with pytest.raises(PermanentIngestionError) as caught:
        await extractor().extract(str(path), "application/pdf")

    assert caught.value.code == "PDF_UNREADABLE"


async def test_a_pdf_over_the_page_limit_is_refused_before_any_page_is_read(tmp_path):
    engine = FakeOcr()

    with pytest.raises(PermanentIngestionError) as caught:
        await extractor(engine, max_pages=2).extract(
            native_pdf(tmp_path, pages=3), "application/pdf"
        )

    assert caught.value.code == "ASSET_TOO_MANY_PAGES"
    assert engine.calls == 0


async def test_a_document_without_any_text_fails(tmp_path):
    with pytest.raises(PermanentIngestionError) as caught:
        await extractor(FakeOcr([])).extract(scanned_pdf(tmp_path), "application/pdf")

    assert caught.value.code == "EXTRACTION_EMPTY"


async def test_an_unreadable_image_upload_fails_permanently(tmp_path):
    image = tmp_path / "photo.png"
    image.write_bytes(b"\x89PNG not really")

    with pytest.raises(PermanentIngestionError) as caught:
        await extractor(FakeOcr(error=OCRExtractionError("truncated"))).extract(
            str(image), "image/png"
        )

    assert caught.value.code == "IMAGE_UNREADABLE"


async def test_an_unavailable_ocr_engine_is_a_pipeline_failure(tmp_path):
    def unavailable():
        raise OCRDependencyUnavailableError("paddleocr", RuntimeError("no models"))

    with pytest.raises(PipelineUnavailableError):
        await extractor(factory=unavailable).extract(scanned_pdf(tmp_path), "application/pdf")


async def test_extraction_has_one_deadline(tmp_path):
    with pytest.raises(PermanentIngestionError) as caught:
        await extractor(FakeOcr(delay=1.0), timeout=0.1).extract(
            scanned_pdf(tmp_path), "application/pdf"
        )

    assert caught.value.code == "EXTRACTION_TIMEOUT"


async def test_the_deadline_waits_for_a_page_read_in_flight_before_cleanup(tmp_path, monkeypatch):
    """The deadline can pass while a worker thread still reads a page from
    the open document and renders it into the temporary directory. The
    extraction ends with EXTRACTION_TIMEOUT only once that thread is done,
    so the directory is never removed (nor the document closed) under it.
    Without the wait, the cleanup raced the render: rmdir failed with
    ENOTEMPTY and that OSError replaced the timeout (seen 2 in 3000 runs
    of the test above)."""
    import tempfile
    import threading
    import time

    from app.services.ingestion import extraction

    finished = threading.Event()
    renders: list[str] = []
    real_read_page = extraction.DocumentExtractor._read_page

    def slow_read_page(self, pdf, index, tmp):
        time.sleep(0.4)  # still running when the 0.1 s deadline passes
        result = real_read_page(self, pdf, index, tmp)  # renders into tmp
        renders.append(tmp)
        finished.set()
        return result

    monkeypatch.setattr(extraction.DocumentExtractor, "_read_page", slow_read_page)
    pages_dir = tmp_path / "tmp"
    pages_dir.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(pages_dir))

    with pytest.raises(PermanentIngestionError) as caught:
        await extractor(FakeOcr(), timeout=0.1).extract(scanned_pdf(tmp_path), "application/pdf")

    assert caught.value.code == "EXTRACTION_TIMEOUT"
    assert finished.is_set()  # the read completed before the extraction ended
    assert list(pages_dir.iterdir()) == []  # its directory, and the render, are gone


def fail_page_dir_removal(monkeypatch):
    """The real failure, in the real code path: a file appears in the page
    directory after rmtree has emptied it, so its rmdir fails with ENOTEMPTY."""
    import os

    real_rmdir = os.rmdir

    def racing_rmdir(path, *, dir_fd=None):
        if "pages-" in os.fspath(path):
            late = os.path.join(os.fspath(path), "late.png")
            os.close(os.open(late, os.O_CREAT | os.O_WRONLY, 0o600, dir_fd=dir_fd))
        return real_rmdir(path, dir_fd=dir_fd)

    monkeypatch.setattr(os, "rmdir", racing_rmdir)


async def test_a_cleanup_failure_never_replaces_the_timeout(tmp_path, monkeypatch, caplog):
    fail_page_dir_removal(monkeypatch)
    with pytest.raises(PermanentIngestionError) as caught:
        await extractor(FakeOcr(delay=1.0), timeout=0.1).extract(
            scanned_pdf(tmp_path), "application/pdf"
        )
    assert caught.value.code == "EXTRACTION_TIMEOUT"
    assert "Page directory not removed after cancellation" in caplog.text  # reported, not hidden


async def test_a_cleanup_failure_without_cancellation_is_not_suppressed(tmp_path, monkeypatch):
    fail_page_dir_removal(monkeypatch)
    with pytest.raises(OSError, match="not empty"):
        await extractor(FakeOcr()).extract(scanned_pdf(tmp_path), "application/pdf")


async def test_an_outside_cancellation_also_waits_for_the_page_read(tmp_path, monkeypatch):
    """The worker fencing a task cancels extraction the same way."""
    import threading
    import time

    from app.services.ingestion import extraction

    started, finished = threading.Event(), threading.Event()
    real_read_page = extraction.DocumentExtractor._read_page

    def slow_read_page(self, pdf, index, tmp):
        started.set()
        time.sleep(0.3)
        result = real_read_page(self, pdf, index, tmp)
        finished.set()
        return result

    monkeypatch.setattr(extraction.DocumentExtractor, "_read_page", slow_read_page)
    running = asyncio.create_task(
        extractor(FakeOcr()).extract(scanned_pdf(tmp_path), "application/pdf")
    )
    while not started.is_set():
        await asyncio.sleep(0.01)
    running.cancel()
    with pytest.raises(asyncio.CancelledError):
        await running
    assert finished.is_set()


@pytest.mark.parametrize(
    "timeout,first_cancel_from_outside",
    [(0.1, False), (30.0, True)],
    ids=["deadline-then-cancel", "cancel-then-cancel"],
)
async def test_a_second_cancellation_cannot_end_the_wait_for_a_page_read(
    tmp_path, monkeypatch, timeout, first_cancel_from_outside
):
    """The document and the page directory stay alive until the render ends,
    however many cancellations arrive while waiting for it."""
    import os
    import threading

    from app.services.ingestion import extraction

    started, release, finished = threading.Event(), threading.Event(), threading.Event()
    seen: dict = {}
    real_read_page = extraction.DocumentExtractor._read_page

    def blocked_read_page(self, pdf, index, tmp):
        started.set()
        assert release.wait(10)
        seen["directory_alive"] = os.path.isdir(tmp)
        seen["document_open"] = not pdf.is_closed
        result = real_read_page(self, pdf, index, tmp)
        finished.set()
        return result

    monkeypatch.setattr(extraction.DocumentExtractor, "_read_page", blocked_read_page)
    pages_dir = tmp_path / "pages"
    pages_dir.mkdir()
    document = scanned_pdf(tmp_path)
    monkeypatch.setattr(extraction.tempfile, "tempdir", str(pages_dir))
    running = asyncio.create_task(
        extractor(FakeOcr(), timeout=timeout).extract(document, "application/pdf")
    )
    while not started.is_set():
        await asyncio.sleep(0.01)

    if first_cancel_from_outside:
        running.cancel()
    await asyncio.sleep(0.2)  # the deadline, or the first cancel, is now being handled
    for _ in range(2):  # further cancellations, while the wait is active
        running.cancel()
        await asyncio.sleep(0.05)
    assert not running.done()  # still waiting for the render
    assert not finished.is_set()
    assert len(list(pages_dir.iterdir())) == 1  # the directory is still there

    release.set()
    # A cancellation from outside stays a cancellation, as asyncio.timeout
    # defines: it is not turned into the deadline's failure.
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(running, timeout=10)
    assert finished.is_set()
    assert seen == {"directory_alive": True, "document_open": True}
    assert list(pages_dir.iterdir()) == []  # cleanup completed afterwards
