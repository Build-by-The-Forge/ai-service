from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from app.core.documents.config import OCRConfig
from app.core.documents.contracts import Document, DocumentPage, OCRDocument, OCRPage


class OCRPort(ABC):
    """
    Abstract Port defining the OCR and Document Extraction boundary.
    Decouples document processing from concrete OpenVINO, PaddleOCR, or Cloud OCR engines.
    """

    @abstractmethod
    async def extract_page(
        self,
        page: DocumentPage,
        config: Optional[OCRConfig] = None,
    ) -> OCRPage:
        """
        Extract text, bounding boxes, block types, and confidence scores from a single DocumentPage.
        """
        pass

    @abstractmethod
    async def extract_document(
        self,
        document: Document,
        config: Optional[OCRConfig] = None,
    ) -> OCRDocument:
        """
        Extract and aggregate OCR structures across all pages of a Document.
        """
        pass

    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """
        Verifies OCR runtime engine readiness and compiled model states.
        """
        pass
