from abc import ABC, abstractmethod
from typing import List, Dict


class OCRProvider(ABC):

    @abstractmethod
    async def extract_text_from_image(self, image_path: str) -> List[Dict]:
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
        pass
