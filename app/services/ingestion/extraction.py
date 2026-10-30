"""Text extraction from a verified document (P3-2).

Every PDF page is classified from PyMuPDF measurements, then read the
cheapest way that is accurate (docs/p3-2-extraction-baseline.md):

- native: the page has a text layer (at least ``NATIVE_MIN_CHARS``
  characters) and images cover less than half of it. Its text layer is
  read in milliseconds and beats OCR by a wide margin, especially on
  mathematics. The reading order is chosen per page: LaTeX output is drawn
  in reading order and sorting would split its formulas, while office
  exports draw in arbitrary order and must be sorted by position. A page
  is sorted when its drawing order often jumps upward
  (``SORT_WHEN_JUMPING``). Typographic ligatures are expanded.
- mixed: a text layer and images over half the page. The text layer is
  read, with a warning that text inside the images is not.
- scanned / image: rendered at 2x and OCR'd.
- empty: no text layer and no images. Reported, with no text.

Images (PNG, JPEG) are OCR'd directly. The OCR engine is created only when
a page needs it, so native documents never load the model. Rendering and
OCR run in worker threads; one deadline covers the whole document.

Failures are typed (``ingestion_outcomes``):
- permanent: PDF_ENCRYPTED, PDF_UNREADABLE, ASSET_TOO_MANY_PAGES,
  IMAGE_UNREADABLE, EXTRACTION_EMPTY, EXTRACTION_TIMEOUT;
- pipeline unavailable (INGESTION_PIPELINE_UNAVAILABLE): the OCR engine
  cannot load in this deployment.
"""

from __future__ import annotations

import asyncio
import logging
import os
import statistics
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

import fitz

from app.core.documents.errors import OCRDependencyUnavailableError, OCRExtractionError
from app.services.tasks.ingestion_outcomes import (
    IngestionFailure,
    PermanentIngestionError,
    PipelineUnavailableError,
)

logger = logging.getLogger(__name__)

NATIVE_MIN_CHARS = 200
SCANNED_IMAGE_COVERAGE = 0.5
# A jump counts when the next drawn block starts this fraction of the page
# height above the previous one; a page is sorted when at least
# SORT_WHEN_JUMPING of its consecutive blocks jump. Measured: LaTeX pages
# 0.00, office exports 0.12 and 0.29.
UPWARD_JUMP = 0.10
SORT_WHEN_JUMPING = 0.05
RENDER_ZOOM = fitz.Matrix(2, 2)
_LIGATURES = str.maketrans({"ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl", "ﬆ": "st"})

NATIVE, MIXED, SCANNED, EMPTY, IMAGE = "native", "mixed", "scanned", "empty", "image"


class OcrEngine(Protocol):
    async def extract_text_from_image(self, image_path: str) -> list[dict]: ...


@dataclass(frozen=True, slots=True)
class ExtractedPage:
    number: int
    kind: str
    method: str  # "text_layer", "ocr" or "none"
    text: str
    confidence: float | None = None  # mean OCR line confidence
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ExtractedDocument:
    pages: tuple[ExtractedPage, ...]
    seconds: float
    low_confidence_pages: int = 0
    _ocr_confidences: tuple[float, ...] = field(default=(), repr=False)

    @property
    def characters(self) -> int:
        return sum(len(page.text) for page in self.pages)

    def stage_metrics(self) -> dict[str, Any]:
        """Evidence for the "extracted" stage. The backend requires
        characters >= 1, pages, and method "native_text" or "ocr"."""
        ocr_pages = sum(1 for page in self.pages if page.method == "ocr")
        return {
            "characters": self.characters,
            "pages": len(self.pages),
            "method": "ocr" if ocr_pages else "native_text",
            "native_pages": sum(1 for page in self.pages if page.method == "text_layer"),
            "ocr_pages": ocr_pages,
            "empty_pages": sum(1 for page in self.pages if page.method == "none"),
            "mean_ocr_confidence": (
                round(statistics.fmean(self._ocr_confidences), 4) if self._ocr_confidences else None
            ),
            "low_confidence_pages": self.low_confidence_pages,
            "seconds": round(self.seconds, 3),
        }


def _image_coverage(page: fitz.Page) -> float:
    area = page.rect.width * page.rect.height or 1.0
    covered = sum(fitz.Rect(info["bbox"]).get_area() for info in page.get_image_info())
    return min(covered / area, 1.0)


def classify_page(page: fitz.Page) -> str:
    chars = len(page.get_text("text").strip())
    coverage = _image_coverage(page)
    if chars >= NATIVE_MIN_CHARS:
        return MIXED if coverage >= SCANNED_IMAGE_COVERAGE else NATIVE
    if coverage > 0:
        return SCANNED
    return NATIVE if chars else EMPTY


def jump_fraction(page: fitz.Page) -> float:
    """How often the drawing order jumps up the page, block to block."""
    blocks = [b for b in page.get_text("blocks") if b[6] == 0 and b[4].strip()]
    if len(blocks) < 2:
        return 0.0
    step = UPWARD_JUMP * page.rect.height
    jumps = sum(1 for a, b in zip(blocks, blocks[1:]) if b[1] < a[1] - step)
    return jumps / (len(blocks) - 1)


def text_layer(page: fitz.Page) -> str:
    sort = jump_fraction(page) >= SORT_WHEN_JUMPING
    return page.get_text("text", sort=sort).translate(_LIGATURES).strip()


class DocumentExtractor:
    def __init__(
        self,
        *,
        ocr_engine_factory: Callable[[], OcrEngine],
        max_pages: int,
        timeout_seconds: float,
        min_confidence: float,
    ) -> None:
        self._ocr_engine_factory = ocr_engine_factory
        self._engine: OcrEngine | None = None
        self._max_pages = max_pages
        self._timeout_seconds = timeout_seconds
        self._min_confidence = min_confidence

    @classmethod
    def from_settings(cls) -> DocumentExtractor:
        from app.core.config import settings
        from app.services.ocr.paddle import PaddleOCRService

        return cls(
            ocr_engine_factory=PaddleOCRService,
            max_pages=settings.EXTRACTION_MAX_PAGES,
            timeout_seconds=settings.EXTRACTION_TIMEOUT_SECONDS,
            min_confidence=settings.OCR_MIN_MEAN_CONFIDENCE,
        )

    async def extract(self, path: str, mime_type: str) -> ExtractedDocument:
        started = time.monotonic()
        try:
            async with asyncio.timeout(self._timeout_seconds):
                if mime_type == "application/pdf":
                    pages = await self._pdf(path)
                else:
                    pages = [await self._image(path)]
        except TimeoutError:
            raise PermanentIngestionError(
                "EXTRACTION_TIMEOUT",
                f"extraction did not finish within {self._timeout_seconds:g} s",
            ) from None
        confidences = tuple(p.confidence for p in pages if p.confidence is not None)
        document = ExtractedDocument(
            pages=tuple(pages),
            seconds=time.monotonic() - started,
            low_confidence_pages=sum(1 for c in confidences if c < self._min_confidence),
            _ocr_confidences=confidences,
        )
        if document.characters == 0:
            raise PermanentIngestionError("EXTRACTION_EMPTY", "no text could be extracted")
        return document

    async def _pdf(self, path: str) -> list[ExtractedPage]:
        try:
            pdf = await asyncio.to_thread(fitz.open, path)
        except Exception as exc:
            raise PermanentIngestionError(
                "PDF_UNREADABLE", f"the PDF cannot be read ({type(exc).__name__})"
            ) from None
        with pdf:
            # Only a required password refuses the file: permission-only
            # encryption (common in course packs) opens and reads normally.
            if pdf.needs_pass:
                raise PermanentIngestionError("PDF_ENCRYPTED", "the PDF is password-protected")
            if len(pdf) > self._max_pages:
                raise PermanentIngestionError(
                    "ASSET_TOO_MANY_PAGES",
                    f"the PDF has {len(pdf)} pages; the limit is {self._max_pages}",
                )
            if len(pdf) == 0:
                raise PermanentIngestionError("PDF_UNREADABLE", "the PDF has no pages")
            pages = []
            try:
                with tempfile.TemporaryDirectory(prefix="pages-") as tmp:
                    for index in range(len(pdf)):
                        pages.append(await self._pdf_page(pdf, index, tmp))
            except OSError as exc:
                # Only a cleanup failure during cancellation: it must not
                # replace the cancellation, or the deadline is reported as
                # an unclassified OSError. Any other OSError still propagates.
                if not isinstance(exc.__context__, asyncio.CancelledError):
                    raise
                logger.warning("Page directory not removed after cancellation: %s", exc)
                raise exc.__context__ from None
            return pages

    async def _pdf_page(self, pdf: fitz.Document, index: int, tmp: str) -> ExtractedPage:
        number = index + 1
        read = asyncio.ensure_future(asyncio.to_thread(self._read_page, pdf, index, tmp))
        try:
            kind, text, image = await asyncio.shield(read)
        except asyncio.CancelledError:
            # The deadline arrived while a worker thread reads this page from
            # the open document and may still render it into tmp. A thread
            # cannot be interrupted: wait for it (one page, bounded) before
            # the caller closes the document and removes tmp. Otherwise the
            # cleanup races the render: the directory removal fails with
            # ENOTEMPTY, that OSError replaces the timeout, and the document
            # is closed under a thread still using it.
            await asyncio.wait({read})
            raise
        except IngestionFailure:
            raise
        except Exception as exc:  # a damaged page: reported, the rest continue
            return ExtractedPage(
                number, EMPTY, "none", "", warnings=(f"page_unreadable:{type(exc).__name__}",)
            )
        if image is None:
            warnings = ("images_not_read",) if kind == MIXED else ()
            if kind == EMPTY:
                return ExtractedPage(number, kind, "none", "", warnings=("empty_page",))
            return ExtractedPage(number, kind, "text_layer", text, warnings=warnings)
        try:
            return await self._ocr_page(number, kind, image)
        finally:
            os.unlink(image)

    def _read_page(self, pdf: fitz.Document, index: int, tmp: str) -> tuple[str, str, str | None]:
        """In a worker thread: classify, then read the text layer or render
        the page for OCR (returns the image path)."""
        page = pdf[index]
        kind = classify_page(page)
        if kind in (NATIVE, MIXED):
            return kind, text_layer(page), None
        if kind == EMPTY:
            return kind, "", None
        image = os.path.join(tmp, f"page-{index + 1}.png")
        page.get_pixmap(matrix=RENDER_ZOOM).save(image)
        return kind, "", image

    async def _image(self, path: str) -> ExtractedPage:
        try:
            return await self._ocr_page(1, IMAGE, path)
        except PermanentIngestionError as failure:
            if failure.code == "OCR_PAGE_FAILED":
                raise PermanentIngestionError("IMAGE_UNREADABLE", failure.message) from None
            raise

    async def _ocr_page(self, number: int, kind: str, image: str) -> ExtractedPage:
        engine = self._ocr_engine()
        try:
            lines = await engine.extract_text_from_image(image)
        except OCRExtractionError as exc:
            if kind == IMAGE:
                raise PermanentIngestionError("OCR_PAGE_FAILED", str(exc)) from None
            return ExtractedPage(number, kind, "ocr", "", warnings=("ocr_failed",))
        text = "\n".join(line["text"] for line in lines).strip()
        confidence = (
            round(statistics.fmean(line["confidence"] for line in lines), 4) if lines else None
        )
        warnings = []
        if not text:
            warnings.append("empty_page")
        elif confidence is not None and confidence < self._min_confidence:
            warnings.append("low_confidence")
        return ExtractedPage(number, kind, "ocr", text, confidence, tuple(warnings))

    def _ocr_engine(self) -> OcrEngine:
        # Loaded on the first page that needs OCR: native documents never
        # pay for the model.
        if self._engine is None:
            try:
                self._engine = self._ocr_engine_factory()
            except OCRDependencyUnavailableError as exc:
                raise PipelineUnavailableError(f"OCR is not available: {exc}") from None
        return self._engine
