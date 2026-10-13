import json
import logging
from pathlib import Path

from app.core.config import STRICT_ENVIRONMENTS, settings
from app.core.documents.config import OCRConfig
from app.core.documents.errors import OCRDependencyUnavailableError, OCRExtractionError
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
                **model_directories(settings.OCR_MODEL_DIR),
            )
        except OCRDependencyUnavailableError:
            raise
        except Exception as exc:  # model load, native runtime
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
        validate_image(image_path)

        # Run OCR on image
        # result is a list of lists: [[text, score], [bbox], ...]
        try:
            result = self.model.ocr(image_path, cls=True)
        except Exception as exc:  # native decoder/runtime; the cause is kept
            raise OCRExtractionError(f"{type(exc).__name__}: {exc}") from exc

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


# Decoders Pillow may use for OCR input; the content decides, not the
# file name. Anything else is rejected before it reaches a decoder.
ALLOWED_IMAGE_FORMATS = ("PNG", "JPEG", "TIFF", "WEBP")


def validate_image(image_path: str, max_pixels: int | None = None) -> None:
    """Reject OCR input that is not a complete image of an allowed format and
    size, with OCRExtractionError, before the OCR engine decodes it.

    The pixel limit is checked from the header, before any decoding, so an
    oversized image (a decompression bomb) is refused without allocating it.
    """
    try:
        from PIL import Image
    except Exception as exc:  # absent, or failing inside the package
        raise OCRDependencyUnavailableError(
            "Pillow", exc, install_hint="pip install Pillow", module="PIL"
        ) from exc

    limit = OCRConfig().max_image_pixels if max_pixels is None else max_pixels
    try:
        with Image.open(image_path, formats=ALLOWED_IMAGE_FORMATS) as image:
            width, height = image.size
            if width * height > limit:
                raise OCRExtractionError(
                    f"image is {width}x{height} ({width * height} pixels), "
                    f"over the {limit} pixel limit"
                )
            image.load()  # full decode: truncated or corrupt data fails here
    except OCRExtractionError:
        raise
    except Image.DecompressionBombError as exc:  # Pillow's own, far larger, limit
        raise OCRExtractionError(f"image is over the {limit} pixel limit: {exc}") from exc
    except Exception as exc:  # unidentified, truncated, corrupt
        raise OCRExtractionError(
            f"not a valid {'/'.join(ALLOWED_IMAGE_FORMATS)} image: {type(exc).__name__}: {exc}"
        ) from exc


MODEL_FILES = ("inference.pdmodel", "inference.pdiparams", "inference.pdiparams.info")
ROLES = {"det": "det_model_dir", "rec": "rec_model_dir", "cls": "cls_model_dir"}


def model_directories(model_dir: str | None) -> dict[str, str]:
    """PaddleOCR model directory arguments for the verified models installed by
    scripts/fetch_ocr_models.py (Phase 2 Step 2).

    With all three directories given and complete, paddleocr performs no
    download. Without OCR_MODEL_DIR, staging/production refuse; development
    and testing fall back to paddleocr's own cache, loudly.
    """
    if not model_dir:
        if settings.APP_ENVIRONMENT in STRICT_ENVIRONMENTS:
            cause = FileNotFoundError("OCR_MODEL_DIR is not set")
            raise OCRDependencyUnavailableError(
                "PaddleOCR models (OCR_MODEL_DIR is required; models are never downloaded at runtime)",
                cause,
            ) from cause
        logger.warning("OCR_MODEL_DIR is not set: paddleocr may download models into its cache")
        return {}

    root = Path(model_dir)
    record = root / "installed.json"
    try:
        installed = json.loads(record.read_text(encoding="utf-8"))["models"]
        directories = {arg: root / role / installed[role]["name"] for role, arg in ROLES.items()}
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise OCRDependencyUnavailableError(
            f"PaddleOCR models in {root} ({record.name} unreadable)", exc
        ) from exc

    for directory in directories.values():
        for name in MODEL_FILES:
            path = directory / name
            if not path.is_file():
                cause = FileNotFoundError(f"missing model file {path}")
                raise OCRDependencyUnavailableError(f"PaddleOCR models in {root}", cause) from cause
    return {arg: str(directory) for arg, directory in directories.items()}
