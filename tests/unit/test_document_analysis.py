import pytest

from app.core.documents import (
    Document,
    DocumentAnalysisClassification,
    DocumentAnalysisResult,
    DocumentAnalyzer,
    DocumentMetadata,
    DocumentSource,
    DocumentStatus,
    DocumentType,
    PageAnalysis,
    PageContentType,
    UnsupportedDocumentTypeError,
)
from app.core.rag.contracts import ContentAuthority
from app.core.tenant import clear_request_scope


@pytest.fixture(autouse=True)
def reset_tenant_context():
    try:
        yield
    finally:
        clear_request_scope()


def _make_document(*, mime_type: str, title: str = "Sample", pages=None):
    metadata = DocumentMetadata(
        filename=f"{title}.pdf" if mime_type == "application/pdf" else f"{title}.png",
        mime_type=mime_type,
        byte_size=1024,
        content_sha256="a" * 64,
        storage_uri=f"tenants/42/documents/{title}",
        authority=ContentAuthority.INSTITUTIONAL,
    )
    doc = Document.create(
        tenant_id="42",
        title=title,
        metadata=metadata,
        doc_type=DocumentType.PDF if mime_type == "application/pdf" else DocumentType.IMAGE,
        source=DocumentSource.UPLOAD,
    )
    doc.pages = pages or []
    doc.status = DocumentStatus.RECEIVED
    return doc


def test_native_pdf_is_classified_as_native_text():
    page = PageAnalysis(
        page_number=1,
        content_type=PageContentType.NATIVE_TEXT,
        has_text=True,
        has_images=False,
        text_density=0.8,
        reason="native text page",
    )
    doc = _make_document(mime_type="application/pdf", pages=[page])

    result = DocumentAnalyzer().analyze_document(doc)

    assert result.document_type == DocumentType.PDF
    assert result.classification == DocumentAnalysisClassification.NATIVE_TEXT
    assert result.next_stage == "text_normalization"
    assert result.pages[0].content_type == PageContentType.NATIVE_TEXT


def test_scanned_pdf_is_classified_as_scanned():
    page = PageAnalysis(
        page_number=1,
        content_type=PageContentType.SCANNED,
        has_text=False,
        has_images=True,
        text_density=0.05,
        reason="scanned page without native text",
    )
    doc = _make_document(mime_type="application/pdf", pages=[page])

    result = DocumentAnalyzer().analyze_document(doc)

    assert result.classification == DocumentAnalysisClassification.SCANNED
    assert result.next_stage == "ocr_required"


def test_mixed_pdf_is_classified_as_mixed():
    doc = _make_document(
        mime_type="application/pdf",
        pages=[
            PageAnalysis(
                page_number=1,
                content_type=PageContentType.NATIVE_TEXT,
                has_text=True,
                has_images=False,
                text_density=0.75,
                reason="native",
            ),
            PageAnalysis(
                page_number=2,
                content_type=PageContentType.SCANNED,
                has_text=False,
                has_images=True,
                text_density=0.02,
                reason="scanned",
            ),
        ],
    )

    result = DocumentAnalyzer().analyze_document(doc)

    assert result.classification == DocumentAnalysisClassification.MIXED
    assert result.next_stage == "ocr_required"
    assert [p.page_number for p in result.pages if p.content_type == PageContentType.SCANNED] == [2]


def test_raster_image_is_classified_for_ocr_path():
    doc = _make_document(
        mime_type="image/png",
        pages=[
            PageAnalysis(
                page_number=1,
                content_type=PageContentType.SCANNED,
                has_text=False,
                has_images=True,
                text_density=0.0,
                reason="raster image",
            )
        ],
    )

    result = DocumentAnalyzer().analyze_document(doc)

    assert result.document_type == DocumentType.IMAGE
    assert result.classification == DocumentAnalysisClassification.IMAGE
    assert result.next_stage == "ocr_required"


def test_unsupported_mime_is_rejected():
    doc = _make_document(mime_type="text/plain")

    with pytest.raises(UnsupportedDocumentTypeError):
        DocumentAnalyzer().analyze_document(doc)


def test_analysis_result_is_structured_and_repeatable():
    doc = _make_document(
        mime_type="application/pdf",
        pages=[
            PageAnalysis(
                page_number=1,
                content_type=PageContentType.NATIVE_TEXT,
                has_text=True,
                has_images=False,
                text_density=0.9,
                reason="ordered text",
            ),
            PageAnalysis(
                page_number=2,
                content_type=PageContentType.NATIVE_TEXT,
                has_text=True,
                has_images=False,
                text_density=0.7,
                reason="ordered text",
            ),
        ],
    )

    result = DocumentAnalyzer().analyze_document(doc)

    assert isinstance(result, DocumentAnalysisResult)
    assert result.page_count == 2
    assert all(isinstance(page, PageAnalysis) for page in result.pages)
    assert result.classification == DocumentAnalysisClassification.NATIVE_TEXT
    assert result.next_stage == "text_normalization"
