from abc import ABC, abstractmethod


class OCRProvider(ABC):
    @abstractmethod
    async def extract_text_from_image(self, image_path: str) -> list[dict]:
        """
        Returns structured OCR output:
        [
            {
                "text": "...",
                "confidence": 0.98,
                "bbox": [...]
            }
        ]
        """
