from abc import ABC, abstractmethod
from typing import Any

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
        config: OCRConfig | None = None,
    ) -> OCRPage:
        """
        Extract text, bounding boxes, block types, and confidence scores from a single DocumentPage.
        """

    @abstractmethod
    async def extract_document(
        self,
        document: Document,
        config: OCRConfig | None = None,
    ) -> OCRDocument:
        """
        Extract and aggregate OCR structures across all pages of a Document.
        """

    @abstractmethod
    async def health_check(self) -> dict[str, Any]:
        """
        Verifies OCR runtime engine readiness and compiled model states.
        """
