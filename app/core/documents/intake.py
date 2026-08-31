import hashlib
import re
from dataclasses import dataclass
from pathlib import PurePosixPath

from app.core.documents.config import OCRConfig
from app.core.documents.contracts import Document, DocumentMetadata
from app.core.documents.enums import DocumentSource, DocumentStatus, DocumentType
from app.core.documents.errors import (
    DocumentSizeLimitExceededError,
    DocumentTenantMissingError,
    UnsupportedDocumentTypeError,
)
from app.core.rag.contracts import ContentAuthority
from app.core.tenant import get_school_id

_SUPPORTED_MIME_TYPES = {
    "application/pdf": DocumentType.PDF,
    "image/png": DocumentType.IMAGE,
    "image/jpeg": DocumentType.IMAGE,
    "image/tiff": DocumentType.IMAGE,
    "image/webp": DocumentType.IMAGE,
}


class UnsafeFilenameError(ValueError):
    """Raised when a document filename is unsafe or inconsistent with the storage boundary."""


@dataclass(frozen=True)
class DocumentIngestionResult:
    document_id: str
    tenant_id: str
    sha256: str
    status: DocumentStatus
    storage_uri: str
    document: Document


class DocumentStoragePort:
    """Minimal storage port that keeps tenant and document identity isolated by design."""

    async def write_document(
        self, tenant_id: str, document_id: str, filename: str, content: bytes
    ) -> str:
        safe_name = PurePosixPath(filename).name
        if not safe_name or safe_name != filename or "/" in filename or "\\" in filename:
            raise UnsafeFilenameError(f"Unsafe filename: {filename!r}")
        storage_uri = f"tenants/{tenant_id}/documents/{document_id}/{safe_name}"
        return storage_uri


class DocumentIntakeService:
    """Secure intake service for authoritative tenant scoping and document registration."""

    def __init__(
        self,
        max_document_bytes: int = 20 * 1024 * 1024,
        storage_port: DocumentStoragePort | None = None,
        ocr_config: OCRConfig | None = None,
    ):
        self.max_document_bytes = max_document_bytes
        self.storage_port = storage_port or DocumentStoragePort()
        self.ocr_config = ocr_config or OCRConfig()

    def _require_authoritative_tenant(self) -> str:
        tenant_id = str(get_school_id())
        if not tenant_id or tenant_id == "0" or tenant_id == "None":
            raise DocumentTenantMissingError("Document requires an authoritative tenant_id")
        return tenant_id

    def _validate_filename(self, filename: str) -> str:
        if not filename or not filename.strip():
            raise UnsafeFilenameError("Document filename is required")
        if filename.strip() != filename:
            raise UnsafeFilenameError("Document filename contains leading or trailing whitespace")
        if "/" in filename or "\\" in filename:
            raise UnsafeFilenameError("Document filename must not contain path separators")
        if ".." in filename:
            raise UnsafeFilenameError("Document filename must not contain traversal segments")
        if not re.fullmatch(r"[A-Za-z0-9._-]+", PurePosixPath(filename).name):
            raise UnsafeFilenameError(f"Unsafe filename: {filename!r}")
        return PurePosixPath(filename).name

    def _validate_mime(self, mime_type: str) -> DocumentType:
        normalized = mime_type.lower().strip()
        try:
            return _SUPPORTED_MIME_TYPES[normalized]
        except KeyError as exc:
            raise UnsupportedDocumentTypeError(normalized) from exc

    def _calculate_hash(self, content: bytes) -> str:
        return hashlib.sha256(content).hexdigest()

    async def submit_document_async(
        self,
        *,
        filename: str,
        content: bytes,
        mime_type: str,
        source: DocumentSource,
        title: str | None = None,
    ) -> DocumentIngestionResult:
        tenant_id = self._require_authoritative_tenant()
        safe_name = self._validate_filename(filename)
        doc_type = self._validate_mime(mime_type)

        if not content:
            raise ValueError("Document content cannot be empty")
        if len(content) > self.max_document_bytes:
            raise DocumentSizeLimitExceededError(
                f"Document exceeds maximum size of {self.max_document_bytes} bytes",
                max_limit=self.max_document_bytes,
                current_value=len(content),
                tenant_id=tenant_id,
            )

        content_hash = self._calculate_hash(content)
        document_id = f"doc_{hashlib.sha256(f'{tenant_id}:{safe_name}:{content_hash}'.encode()).hexdigest()[:12]}"
        storage_uri = f"tenants/{tenant_id}/documents/{document_id}/{safe_name}"

        metadata = DocumentMetadata(
            filename=safe_name,
            mime_type=mime_type.lower().strip(),
            byte_size=len(content),
            page_count=0,
            content_sha256=content_hash,
            storage_uri=storage_uri,
            authority=ContentAuthority.INSTITUTIONAL,
        )
        document = Document.create(
            tenant_id=tenant_id,
            title=title or safe_name,
            metadata=metadata,
            doc_type=doc_type,
            source=source,
            document_id=document_id,
        )
        document.status = DocumentStatus.RECEIVED
        document.updated_at = __import__("time").time()

        return DocumentIngestionResult(
            document_id=document.document_id,
            tenant_id=document.tenant_id,
            sha256=content_hash,
            status=document.status,
            storage_uri=storage_uri,
            document=document,
        )

    def submit_document(
        self,
        *,
        filename: str,
        content: bytes,
        mime_type: str,
        source: DocumentSource,
        title: str | None = None,
    ) -> DocumentIngestionResult:
        return __import__("asyncio").run(
            self.submit_document_async(
                filename=filename,
                content=content,
                mime_type=mime_type,
                source=source,
                title=title,
            )
        )
