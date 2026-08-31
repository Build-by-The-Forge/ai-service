import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from app.core.documents.enums import (
    DocumentSource,
    DocumentStatus,
    DocumentType,
    OCRBlockType,
    PageType,
)
from app.core.documents.errors import (
    DocumentTenantMissingError,
    InvalidBoundingBoxError,
)
from app.core.rag.contracts import ChunkProvenance, ContentAuthority


@dataclass(frozen=True)
class OCRBoundingBox:
    """
    Geometric coordinate representation of an extracted text or structural region.
    Coordinates can be pixel-space or normalized [0.0, 1.0].
    """

    x_min: float
    y_min: float
    x_max: float
    y_max: float
    polygon: list[list[float]] | None = None

    def __post_init__(self):
        if self.x_max < self.x_min or self.y_max < self.y_min:
            raise InvalidBoundingBoxError(
                f"Invalid bounding box dimensions: [{self.x_min}, {self.y_min}, {self.x_max}, {self.y_max}]"
            )

    @property
    def width(self) -> float:
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        return self.y_max - self.y_min

    @property
    def area(self) -> float:
        return self.width * self.height

    def to_dict(self) -> dict[str, Any]:
        return {
            "x_min": round(self.x_min, 4),
            "y_min": round(self.y_min, 4),
            "x_max": round(self.x_max, 4),
            "y_max": round(self.y_max, 4),
            "polygon": self.polygon,
        }


@dataclass
class OCRBlock:
    """
    Structural unit of OCR or native-text extraction (heading, paragraph, equation, table).
    """

    block_id: str
    page_number: int
    text: str
    confidence: float
    bounding_box: OCRBoundingBox
    block_type: OCRBlockType = OCRBlockType.PARAGRAPH
    reading_order: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        page_number: int,
        text: str,
        confidence: float,
        bounding_box: OCRBoundingBox,
        block_type: OCRBlockType = OCRBlockType.PARAGRAPH,
        reading_order: int = 1,
        metadata: dict[str, Any] | None = None,
    ) -> "OCRBlock":
        return cls(
            block_id=f"blk_{uuid.uuid4().hex[:12]}",
            page_number=page_number,
            text=text.strip(),
            confidence=max(0.0, min(1.0, confidence)),
            bounding_box=bounding_box,
            block_type=block_type,
            reading_order=reading_order,
            metadata=metadata or {},
        )


@dataclass
class DocumentPage:
    """
    Single page representation within a multi-page document.
    """

    page_number: int
    page_type: PageType = PageType.NATIVE_TEXT
    raw_text: str | None = None
    width: int | None = None
    height: int | None = None
    has_native_text: bool = False
    blocks: list[OCRBlock] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class OCRPage:
    """
    Normalized result of OCR extraction for a single page.
    """

    page_number: int
    blocks: list[OCRBlock]
    confidence: float = 1.0
    full_text: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class OCRDocument:
    """
    Aggregated OCR extraction across all pages of a document.
    """

    document_id: str
    tenant_id: str
    pages: list[OCRPage]
    overall_confidence: float = 1.0
    total_blocks: int = 0
    full_text: str = ""


@dataclass
class DocumentMetadata:
    """
    Header, source, and cryptographic provenance metadata for an ingested document.
    """

    filename: str
    mime_type: str
    byte_size: int
    page_count: int = 0
    course_code: str | None = None
    topic: str | None = None
    authority: ContentAuthority = ContentAuthority.INSTITUTIONAL
    storage_uri: str | None = None
    content_sha256: str | None = None
    custom_attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class Document:
    """
    Authoritative canonical Document entity within the BUILD AI Service.
    Strictly isolated by authoritative tenant_id.
    """

    document_id: str
    tenant_id: str
    title: str
    doc_type: DocumentType
    source: DocumentSource
    status: DocumentStatus
    metadata: DocumentMetadata
    pages: list[DocumentPage] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def __post_init__(self):
        if not self.tenant_id or not str(self.tenant_id).strip():
            raise DocumentTenantMissingError("Document must have a valid authoritative tenant_id")

    @classmethod
    def create(
        cls,
        tenant_id: str,
        title: str,
        metadata: DocumentMetadata,
        doc_type: DocumentType = DocumentType.PDF,
        source: DocumentSource = DocumentSource.UPLOAD,
        document_id: str | None = None,
    ) -> "Document":
        return cls(
            document_id=document_id or f"doc_{uuid.uuid4().hex[:12]}",
            tenant_id=tenant_id,
            title=title,
            doc_type=doc_type,
            source=source,
            status=DocumentStatus.PENDING,
            metadata=metadata,
        )

    def to_chunk_provenance(
        self, page_number: int, section_title: str | None = None
    ) -> ChunkProvenance:
        """
        Generates an AI-2A compatible ChunkProvenance record linking back to this document.
        """
        return ChunkProvenance(
            source_uri=self.metadata.storage_uri or f"doc://{self.tenant_id}/{self.document_id}",
            page_number=page_number,
            section_title=section_title,
            authority=self.metadata.authority,
            content_sha256=self.metadata.content_sha256,
        )
