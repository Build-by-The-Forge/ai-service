import logging
from pathlib import Path

from app.core.documents.errors import OCRDependencyUnavailableError
from app.services.ocr.base import OCRProvider

logger = logging.getLogger(__name__)

# The import failure is kept, not replaced: paddleocr loads a large native
# stack, and "not installed" was hiding failures that happen *inside* an
# installed package (Phase 2 audit F5).
_IMPORT_ERROR: BaseException | None = None
try:
    from paddleocr import PaddleOCR
except Exception as exc:  # any failure, not only ImportError
    PaddleOCR = None
    _IMPORT_ERROR = exc
    logger.warning("paddleocr import failed: %s: %s", type(exc).__name__, exc, exc_info=exc)


class PaddleOCRService(OCRProvider):
    def __init__(self):
        """Initialize PaddleOCR for real OCR processing."""
        if PaddleOCR is None:
            cause = _IMPORT_ERROR or ModuleNotFoundError(
                "No module named 'paddleocr'", name="paddleocr"
            )
            raise OCRDependencyUnavailableError(
                "paddleocr", cause, install_hint="pip install paddleocr", module="paddleocr"
            ) from cause

        # Use the actual installed PaddleOCR API signature.
        # The library exposes 'lang' and 'ocr_version'; it does not accept use_gpu.
        try:
            self.model = PaddleOCR(
                lang="en",
                ocr_version="PP-OCRv3",
            )
        except Exception as exc:  # model download/load, native runtime
            raise OCRDependencyUnavailableError("PaddleOCR model (en, PP-OCRv3)", exc) from exc

    async def extract_text_from_image(self, image_path: str) -> list[dict]:
        """
        Extract text from image using PaddleOCR.

        Args:
            image_path: Path to the image file

        Returns:
            List of dictionaries with text, confidence, and bbox
        """
        if self.model is None:
            raise RuntimeError("PaddleOCR model not initialized")
        if not Path(image_path).is_file():
            # Checked here, not left to the engine's own error deep inside it.
            raise FileNotFoundError(f"OCR input image not found: {image_path}")

        # Run OCR on image
        # result is a list of lists: [[text, score], [bbox], ...]
        result = self.model.ocr(image_path, cls=True)

        if not result or not result[0]:
            return []

        # Normalize output to expected format
        extracted = []
        for line in result[0]:
            if len(line) >= 2:
                bbox, (text, confidence) = line[0], line[1]
                extracted.append(
                    {
                        "text": text,
                        "confidence": float(confidence),
                        "bbox": [[float(x), float(y)] for x, y in bbox],
                    }
                )

        return extracted
