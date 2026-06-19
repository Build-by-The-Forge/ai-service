from typing import List, Dict
from app.services.ocr.base import OCRProvider


class PaddleOCRService(OCRProvider):

    def __init__(self):
        # In production, replace this placeholder with a real PaddleOCR instance.
        # from paddleocr import PaddleOCR
        # self.model = PaddleOCR(use_angle_cls=True, lang="en")
        self.model = None

    async def extract_text_from_image(self, image_path: str) -> List[Dict]:
        """
        Production flow:
        result = self.model.ocr(image_path)
        """
        return [
            {
                "text": "sample extracted text",
                "confidence": 0.99,
                "bbox": [[0, 0], [100, 0], [100, 20], [0, 20]],
            }
        ]
