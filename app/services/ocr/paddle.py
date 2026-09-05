from app.services.ocr.base import OCRProvider

try:
    from paddleocr import PaddleOCR
except ImportError:
    PaddleOCR = None


class PaddleOCRService(OCRProvider):
    def __init__(self):
        """Initialize PaddleOCR for real OCR processing."""
        if PaddleOCR is None:
            raise RuntimeError("PaddleOCR is not installed. Install with: pip install paddleocr")

        # Use the actual installed PaddleOCR API signature.
        # The library exposes 'lang' and 'ocr_version'; it does not accept use_gpu.
        self.model = PaddleOCR(
            lang="en",
            ocr_version="PP-OCRv3",
        )

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
