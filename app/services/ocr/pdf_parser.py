import logging

from app.core.documents.errors import OCRDependencyUnavailableError

logger = logging.getLogger(__name__)

# Keep the real import failure (Phase 2 audit F5) instead of a generic
# "PyMuPDF is required" later.
_IMPORT_ERROR: BaseException | None = None
try:
    import fitz  # PyMuPDF
except Exception as exc:  # pragma: no cover - depends on the environment
    fitz = None
    _IMPORT_ERROR = exc
    logger.warning("PyMuPDF import failed: %s: %s", type(exc).__name__, exc, exc_info=exc)


class PDFParser:
    def extract_pages(self, pdf_path: str) -> list[str]:
        if fitz is None:
            cause = _IMPORT_ERROR or ModuleNotFoundError("No module named 'fitz'", name="fitz")
            raise OCRDependencyUnavailableError(
                "PyMuPDF", cause, install_hint="pip install PyMuPDF", module="fitz"
            ) from cause

        doc = fitz.open(pdf_path)

        pages: list[str] = []
        for page in doc:
            text = page.get_text("text")
            pages.append(text)

        return pages
