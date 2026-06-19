from typing import List, Dict
from app.services.ocr.base import OCRProvider


class OpenVINOService(OCRProvider):

    def __init__(self):
        self.model = None

    async def extract_text_from_image(self, image_path: str) -> List[Dict]:
        return [
            {
                "text": "openvino extracted text",
                "confidence": 0.95,
                "bbox": [[0, 0], [100, 0], [100, 20], [0, 20]],
            }
        ]
