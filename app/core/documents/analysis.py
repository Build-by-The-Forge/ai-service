from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from app.core.documents.contracts import Document
from app.core.documents.enums import DocumentType
from app.core.documents.errors import UnsupportedDocumentTypeError


class DocumentAnalysisClassification(str, Enum):
    NATIVE_TEXT = "native_text"
    SCANNED = "scanned"
    MIXED = "mixed"
    IMAGE = "image"
    UNSUPPORTED = "unsupported"


class PageContentType(str, Enum):
    NATIVE_TEXT = "native_text"
    SCANNED = "scanned"
    IMAGE = "image"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class PageAnalysis:
    page_number: int
    content_type: PageContentType
    has_text: bool
    has_images: bool
    text_density: float
    reason: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DocumentAnalysisResult:
    document_id: str
    tenant_id: str
    document_type: DocumentType
    classification: DocumentAnalysisClassification
    page_count: int
    pages: list[PageAnalysis]
    next_stage: str
    metadata: dict[str, Any] = field(default_factory=dict)


class DocumentAnalyzer:
    """Deterministic document analysis for AI-2B.3. This phase does not perform OCR."""

    def analyze_document(self, document: Document) -> DocumentAnalysisResult:
        mime_type = (document.metadata.mime_type or "").lower().strip()

        if mime_type == "application/pdf":
            doc_type = DocumentType.PDF
        elif mime_type in {"image/png", "image/jpeg", "image/tiff", "image/webp"}:
            doc_type = DocumentType.IMAGE
        else:
            raise UnsupportedDocumentTypeError(mime_type, tenant_id=document.tenant_id)

        pages = list(document.pages) if document.pages else []
        if not pages:
            pages = [
                PageAnalysis(
                    page_number=1,
                    content_type=PageContentType.UNKNOWN,
                    has_text=False,
                    has_images=False,
                    text_density=0.0,
                    reason="no page metadata available",
                )
            ]

        if doc_type == DocumentType.IMAGE:
            classification = DocumentAnalysisClassification.IMAGE
            next_stage = "ocr_required"
        elif any(page.content_type == PageContentType.SCANNED for page in pages) and any(
            page.content_type == PageContentType.NATIVE_TEXT for page in pages
        ):
            classification = DocumentAnalysisClassification.MIXED
            next_stage = "ocr_required"
        elif any(page.content_type == PageContentType.SCANNED for page in pages):
            classification = DocumentAnalysisClassification.SCANNED
            next_stage = "ocr_required"
        elif any(page.content_type == PageContentType.NATIVE_TEXT for page in pages):
            classification = DocumentAnalysisClassification.NATIVE_TEXT
            next_stage = "text_normalization"
        else:
            classification = DocumentAnalysisClassification.UNSUPPORTED
            next_stage = "unsupported"

        return DocumentAnalysisResult(
            document_id=document.document_id,
            tenant_id=document.tenant_id,
            document_type=doc_type,
            classification=classification,
            page_count=len(pages),
            pages=pages,
            next_stage=next_stage,
            metadata={
                "mime_type": mime_type,
                "page_classifications": [p.content_type.value for p in pages],
            },
        )
