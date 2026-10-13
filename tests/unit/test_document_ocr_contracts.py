from dataclasses import dataclass, replace

import pytest

from app.core.documents.analysis import (
    DocumentAnalysisResult,
    PageAnalysis,
    PageContentType,
)
from app.core.documents.contracts import Document, DocumentMetadata, OCRBoundingBox
from app.core.documents.enums import DocumentSource, DocumentStatus, DocumentType
from app.core.documents.ocr import (
    OCRProvider,
    OCRRequest,
    OCRResult,
    OCRRouter,
    OpenVINORuntime,
    PaddleOCRProvider,
)
from app.core.rag.contracts import ContentAuthority


@dataclass
class RecordingOCRProvider(OCRProvider):
    calls: list = None

    def __post_init__(self):
        if self.calls is None:
            self.calls = []

    @property
    def name(self):
        return "recording"

    def supports_runtime(self, runtime: str) -> bool:
        return runtime in {"openvino", "cpu"}

    async def process(self, request: OCRRequest) -> OCRResult:
        self.calls.append(request)
        return OCRResult(
            document_id=request.document_id,
            tenant_id=request.tenant_id,
            pages=[],
            overall_confidence=1.0,
            engine_version="recording",
            runtime_version="recording",
        )

    async def health_check(self):
        return {"status": "ok"}


@pytest.fixture
def sample_analysis_result():
    metadata = DocumentMetadata(
        filename="sample.pdf",
        mime_type="application/pdf",
        byte_size=2048,
        content_sha256="abc123",
        storage_uri="tenants/42/documents/sample",
        authority=ContentAuthority.INSTITUTIONAL,
    )
    document = Document(
        document_id="doc_123",
        tenant_id="42",
        title="Sample",
        doc_type=DocumentType.PDF,
        source=DocumentSource.UPLOAD,
        status=DocumentStatus.RECEIVED,
        metadata=metadata,
        pages=[],
    )

    result = DocumentAnalysisResult(
        document_id=document.document_id,
        tenant_id=document.tenant_id,
        document_type=document.doc_type,
        classification="scanned",
        page_count=3,
        pages=[
            PageAnalysis(
                page_number=1,
                content_type=PageContentType.NATIVE_TEXT,
                has_text=True,
                has_images=False,
                text_density=0.9,
                reason="native",
            ),
            PageAnalysis(
                page_number=2,
                content_type=PageContentType.SCANNED,
                has_text=False,
                has_images=True,
                text_density=0.1,
                reason="scanned",
            ),
            PageAnalysis(
                page_number=3,
                content_type=PageContentType.SCANNED,
                has_text=False,
                has_images=True,
                text_density=0.08,
                reason="scanned",
            ),
        ],
        next_stage="ocr_required",
    )
    return result


def test_native_text_page_does_not_trigger_ocr(sample_analysis_result):
    provider = RecordingOCRProvider()
    router = OCRRouter(provider=provider)
    result = replace(
        sample_analysis_result,
        pages=[
            PageAnalysis(
                page_number=1,
                content_type=PageContentType.NATIVE_TEXT,
                has_text=True,
                has_images=False,
                text_density=0.9,
                reason="native",
            ),
        ],
        page_count=1,
        next_stage="text_normalization",
    )

    request = router.build_request(result)

    assert request is None
    assert provider.calls == []


def test_scanned_page_routes_to_provider(sample_analysis_result):
    provider = RecordingOCRProvider()
    router = OCRRouter(provider=provider)

    request = router.build_request(sample_analysis_result)

    assert request is not None
    assert [page.page_number for page in request.pages] == [2, 3]
    assert request.provider_name == "recording"


def test_mixed_document_routes_only_ocr_pages(sample_analysis_result):
    provider = RecordingOCRProvider()
    router = OCRRouter(provider=provider)
    result = replace(
        sample_analysis_result,
        pages=[
            PageAnalysis(
                page_number=1,
                content_type=PageContentType.NATIVE_TEXT,
                has_text=True,
                has_images=False,
                text_density=0.8,
                reason="native",
            ),
            PageAnalysis(
                page_number=2,
                content_type=PageContentType.SCANNED,
                has_text=False,
                has_images=True,
                text_density=0.15,
                reason="scanned",
            ),
            PageAnalysis(
                page_number=3,
                content_type=PageContentType.NATIVE_TEXT,
                has_text=True,
                has_images=False,
                text_density=0.75,
                reason="native",
            ),
        ],
        classification="mixed",
        next_stage="ocr_required",
    )

    request = router.build_request(result)

    assert request is not None
    assert [page.page_number for page in request.pages] == [2]


def test_paddle_ocr_provider_normalizes_raw_sdk_output():
    provider = PaddleOCRProvider(runtime=OpenVINORuntime())
    raw = {
        "document_id": "doc_123",
        "tenant_id": "42",
        "pages": [
            {
                "page_number": 2,
                "full_text": "hello world",
                "confidence": 0.91,
                "blocks": [
                    {
                        "text": "hello world",
                        "confidence": 0.91,
                        "bounding_box": [0.0, 0.0, 1.0, 0.3],
                        "page_number": 2,
                    }
                ],
            }
        ],
        "model": {
            "type": "PP-OCRv3",
            "runtime": "openvino",
        },
    }

    result = provider.normalize_output(raw)

    assert isinstance(result, OCRResult)
    assert result.pages[0].page_number == 2
    assert result.pages[0].full_text == "hello world"
    assert result.pages[0].lines[0].text == "hello world"
    assert result.pages[0].lines[0].bounding_box == OCRBoundingBox(0.0, 0.0, 1.0, 0.3)


def test_wrong_runtime_is_rejected():
    provider = PaddleOCRProvider(runtime=OpenVINORuntime())

    with pytest.raises(ValueError):
        provider.validate_runtime("onnxruntime")


def test_wrong_provider_is_rejected():
    with pytest.raises(TypeError):
        OCRRouter(provider="not-a-provider")


# --- Phase 2 Step 1: no fabricated success --------------------------------------------


@pytest.mark.asyncio
async def test_paddle_ocr_provider_process_fails_explicitly():
    from app.core.documents.errors import OCRNotImplementedError
    from app.core.documents.ocr import OCRPageRequest, OCRRequest

    provider = PaddleOCRProvider(runtime=OpenVINORuntime())
    request = OCRRequest(
        document_id="doc_1",
        tenant_id="t1",
        pages=[
            OCRPageRequest(1, "doc://t1/doc_1/page/1", "application/pdf", PageContentType.SCANNED)
        ],
        provider_name="paddleocr",
        runtime_name="openvino",
        model_name="PP-OCRv3",
    )

    with pytest.raises(OCRNotImplementedError) as exc:
        await provider.process(request)
    assert exc.value.error_code == "OCR_NOT_IMPLEMENTED"
    assert "doc_1" in str(exc.value)


@pytest.mark.asyncio
async def test_paddle_ocr_provider_health_reports_it_is_not_implemented():
    health = await PaddleOCRProvider(runtime=OpenVINORuntime()).health_check()
    assert health["status"] == "unavailable"
    assert health["implemented"] is False
