from app.core.documents.errors import OCRNotImplementedError
from app.services.ocr.base import OCRProvider


class OpenVINOService(OCRProvider):
    """OpenVINO OCR provider: NOT IMPLEMENTED.

    It used to return the constant text "openvino extracted text" for any
    image (Phase 2 audit F1). No OpenVINO inference is wired, so it now
    refuses instead of fabricating content.
    """

    def __init__(self):
        self.model = None

    async def extract_text_from_image(self, image_path: str) -> list[dict]:
        raise OCRNotImplementedError(
            "OpenVINOService.extract_text_from_image",
            f"no OpenVINO OCR inference is wired; {image_path!r} was not read",
        )
