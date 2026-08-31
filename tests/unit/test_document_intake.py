import hashlib

import pytest

from app.core.documents import (
    DocumentSizeLimitExceededError,
    DocumentSource,
    DocumentStatus,
    DocumentTenantMissingError,
    UnsupportedDocumentTypeError,
)
from app.core.documents.intake import DocumentIntakeService, UnsafeFilenameError
from app.core.tenant import set_school_id


@pytest.fixture(autouse=True)
def reset_tenant_context():
    try:
        yield
    finally:
        try:
            set_school_id(0)
        except Exception:
            pass


def test_document_intake_uses_trusted_tenant_context_and_hash():
    set_school_id(42)
    service = DocumentIntakeService(max_document_bytes=10 * 1024 * 1024)

    payload = b"%PDF-1.4\nThis is a minimal PDF payload"
    result = service.submit_document(
        filename="syllabus.pdf",
        content=payload,
        mime_type="application/pdf",
        source=DocumentSource.UPLOAD,
    )

    assert result.tenant_id == "42"
    assert result.document_id.startswith("doc_")
    assert result.status == DocumentStatus.RECEIVED
    assert result.sha256 == hashlib.sha256(payload).hexdigest()
    assert "tenants/42/documents/" in result.storage_uri
    assert "syllabus.pdf" in result.storage_uri
    assert result.document.document_id == result.document_id
    assert result.document.tenant_id == "42"


def test_document_intake_requires_authoritative_tenant_context():
    service = DocumentIntakeService(max_document_bytes=10 * 1024 * 1024)

    with pytest.raises(DocumentTenantMissingError):
        service.submit_document(
            filename="missing-tenant.pdf",
            content=b"%PDF-1.4\nno tenant",
            mime_type="application/pdf",
            source=DocumentSource.UPLOAD,
        )


def test_document_intake_rejects_unsafe_filename_and_unsupported_mime():
    set_school_id(7)
    service = DocumentIntakeService(max_document_bytes=10 * 1024 * 1024)

    with pytest.raises(UnsafeFilenameError):
        service.submit_document(
            filename="../private/notes.pdf",
            content=b"%PDF-1.4\nunsafe",
            mime_type="application/pdf",
            source=DocumentSource.UPLOAD,
        )

    with pytest.raises(UnsupportedDocumentTypeError):
        service.submit_document(
            filename="notes.txt",
            content=b"hello world",
            mime_type="text/plain",
            source=DocumentSource.UPLOAD,
        )


def test_document_intake_rejects_oversized_files():
    set_school_id(9)
    service = DocumentIntakeService(max_document_bytes=32)

    with pytest.raises(DocumentSizeLimitExceededError):
        service.submit_document(
            filename="big.pdf",
            content=b"x" * 128,
            mime_type="application/pdf",
            source=DocumentSource.UPLOAD,
        )


def test_document_intake_uses_generated_storage_namespace_and_statuses():
    set_school_id(10)
    service = DocumentIntakeService(max_document_bytes=128)

    result = service.submit_document(
        filename="course-overview.png",
        content=b"\x89PNG\r\n\x1a\n" + b"x" * 32,
        mime_type="image/png",
        source=DocumentSource.LMS_INTEGRATION,
    )

    assert result.status == DocumentStatus.RECEIVED
    assert result.document.status == DocumentStatus.RECEIVED
    assert result.storage_uri.startswith("tenants/10/documents/")
    assert result.storage_uri.endswith("course-overview.png")
    assert result.document.metadata.mime_type == "image/png"
    assert result.document.metadata.byte_size == len(b"\x89PNG\r\n\x1a\n" + b"x" * 32)
