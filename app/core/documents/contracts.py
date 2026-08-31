from dataclasses import dataclass, field
import time
from typing import Any, Dict, List, Optional
import uuid

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
    polygon: Optional[List[List[float]]] = None

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

    def to_dict(self) -> Dict[str, Any]:
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
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        page_number: int,
        text: str,
        confidence: float,
        bounding_box: OCRBoundingBox,
        block_type: OCRBlockType = OCRBlockType.PARAGRAPH,
        reading_order: int = 1,
        metadata: Optional[Dict[str, Any]] = None,
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
    raw_text: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    has_native_text: bool = False
    blocks: List[OCRBlock] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OCRPage:
    """
    Normalized result of OCR extraction for a single page.
    """

    page_number: int
    blocks: List[OCRBlock]
    confidence: float = 1.0
    full_text: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OCRDocument:
    """
    Aggregated OCR extraction across all pages of a document.
    """

    document_id: str
    tenant_id: str
    pages: List[OCRPage]
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
    course_code: Optional[str] = None
    topic: Optional[str] = None
    authority: ContentAuthority = ContentAuthority.INSTITUTIONAL
    storage_uri: Optional[str] = None
    content_sha256: Optional[str] = None
    custom_attributes: Dict[str, Any] = field(default_factory=dict)


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
    pages: List[DocumentPage] = field(default_factory=list)
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
        document_id: Optional[str] = None,
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
        self, page_number: int, section_title: Optional[str] = None
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
