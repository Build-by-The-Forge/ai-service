from app.core.documents.config import OCRConfig
from app.core.documents.contracts import (
    Document,
    DocumentMetadata,
    DocumentPage,
    OCRBlock,
    OCRBoundingBox,
    OCRDocument,
    OCRPage,
)
from app.core.documents.intake import DocumentIngestionResult, DocumentIntakeService, DocumentStoragePort
from app.core.documents.enums import (
    DocumentSource,
    DocumentStatus,
    DocumentType,
    OCRBlockType,
    PageType,
)
from app.core.documents.errors import (
    CorruptDocumentError,
    DocumentError,
    DocumentSizeLimitExceededError,
    DocumentTenantMissingError,
    InvalidBoundingBoxError,
    OCRExtractionError,
    UnsupportedDocumentTypeError,
)
from app.core.documents.ports import OCRPort

__all__ = [
    "Document",
    "DocumentMetadata",
    "DocumentPage",
    "OCRBlock",
    "OCRBoundingBox",
    "OCRDocument",
    "OCRPage",
    "DocumentType",
    "DocumentSource",
    "DocumentStatus",
    "PageType",
    "OCRBlockType",
    "DocumentError",
    "DocumentTenantMissingError",
    "CorruptDocumentError",
    "UnsupportedDocumentTypeError",
    "DocumentSizeLimitExceededError",
    "OCRExtractionError",
    "InvalidBoundingBoxError",
    "OCRPort",
    "OCRConfig",
    "DocumentIngestionResult",
    "DocumentIntakeService",
    "DocumentStoragePort",
]
