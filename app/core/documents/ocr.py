from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from app.core.documents.analysis import DocumentAnalysisResult, PageContentType
from app.core.documents.contracts import OCRBoundingBox
from app.core.documents.errors import OCRNotImplementedError


@dataclass(frozen=True)
class OCRPageRequest:
    page_number: int
    source_uri: str
    mime_type: str
    page_type: PageContentType
    width: int | None = None
    height: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OCRTextLine:
    text: str
    confidence: float
    bounding_box: OCRBoundingBox
    reading_order: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OCRPageResult:
    page_number: int
    lines: list[OCRTextLine]
    full_text: str
    confidence: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OCRResult:
    document_id: str
    tenant_id: str
    pages: list[OCRPageResult]
    overall_confidence: float
    engine_version: str
    runtime_version: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OCRRequest:
    document_id: str
    tenant_id: str
    pages: list[OCRPageRequest]
    provider_name: str
    runtime_name: str
    model_name: str
    config: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


class OCRProvider(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def supports_runtime(self, runtime: str) -> bool:
        pass

    @abstractmethod
    async def process(self, request: OCRRequest) -> OCRResult:
        pass

    @abstractmethod
    async def health_check(self) -> dict[str, Any]:
        pass


class OpenVINORuntime:
    name = "openvino"
    supported_devices = {"CPU", "GPU", "NPU"}

    def validate_device(self, device: str) -> None:
        if device.upper() not in self.supported_devices:
            raise ValueError(f"Unsupported OpenVINO device: {device}")


class PaddleOCRProvider(OCRProvider):
    """BUILD-owned adapter for the PaddleOCR OCR architecture."""

    def __init__(self, runtime: OpenVINORuntime, model_name: str = "PP-OCRv3"):
        self._runtime = runtime
        self._model_name = model_name

    @property
    def name(self) -> str:
        return "paddleocr"

    def supports_runtime(self, runtime: str) -> bool:
        return runtime.lower() in {"openvino", "cpu", "gpu", "npu"}

    def validate_runtime(self, runtime: str) -> None:
        if not self.supports_runtime(runtime):
            raise ValueError(f"Unsupported runtime for PaddleOCR provider: {runtime}")

    async def process(self, request: OCRRequest) -> OCRResult:
        # No inference is wired to this adapter yet (Phase 2 audit F1). It used
        # to return zero pages at confidence 1.0, which read as a successful,
        # empty recognition.
        raise OCRNotImplementedError(
            "PaddleOCRProvider.process",
            f"no OCR inference is wired to this provider ({len(request.pages)} page(s) "
            f"of document {request.document_id} were not recognised)",
        )

    async def health_check(self) -> dict[str, Any]:
        return {
            "status": "unavailable",
            "implemented": False,
            "reason": "PaddleOCRProvider.process has no OCR inference wired",
            "provider": self.name,
            "runtime": self._runtime.name,
        }

    def normalize_output(self, raw: dict[str, Any]) -> OCRResult:
        pages: list[OCRPageResult] = []
        for raw_page in raw.get("pages", []):
            lines: list[OCRTextLine] = []
            for raw_block in raw_page.get("blocks", []):
                bbox = raw_block.get("bounding_box")
                if bbox is None:
                    bbox = [0.0, 0.0, 0.0, 0.0]
                lines.append(
                    OCRTextLine(
                        text=str(raw_block.get("text", "")),
                        confidence=float(raw_block.get("confidence", 0.0)),
                        bounding_box=OCRBoundingBox(
                            x_min=float(bbox[0]),
                            y_min=float(bbox[1]),
                            x_max=float(bbox[2]),
                            y_max=float(bbox[3]),
                        ),
                        reading_order=0,
                    )
                )
            pages.append(
                OCRPageResult(
                    page_number=int(raw_page.get("page_number", 1)),
                    lines=lines,
                    full_text=str(raw_page.get("full_text", "")),
                    confidence=float(raw_page.get("confidence", 0.0)),
                )
            )
        return OCRResult(
            document_id=str(raw.get("document_id", "unknown")),
            tenant_id=str(raw.get("tenant_id", "unknown")),
            pages=pages,
            overall_confidence=float(raw.get("overall_confidence", 1.0)),
            engine_version=str(raw.get("model", {}).get("type", self._model_name)),
            runtime_version=str(raw.get("model", {}).get("runtime", self._runtime.name)),
            metadata={"provider": self.name},
        )


class OCRRouter:
    def __init__(self, provider: Any):
        if not isinstance(provider, OCRProvider):
            raise TypeError("OCRRouter provider must implement OCRProvider")
        self.provider = provider

    def build_request(self, analysis: DocumentAnalysisResult) -> OCRRequest | None:
        if analysis.next_stage != "ocr_required":
            return None

        ocr_pages: list[OCRPageRequest] = []
        for page in analysis.pages:
            if page.content_type in {PageContentType.NATIVE_TEXT, PageContentType.UNKNOWN}:
                continue
            ocr_pages.append(
                OCRPageRequest(
                    page_number=page.page_number,
                    source_uri=f"doc://{analysis.tenant_id}/{analysis.document_id}/page/{page.page_number}",
                    mime_type="application/pdf",
                    page_type=page.content_type,
                    metadata={
                        "reason": page.reason,
                        "has_text": page.has_text,
                        "has_images": page.has_images,
                    },
                )
            )

        if not ocr_pages:
            return None

        return OCRRequest(
            document_id=analysis.document_id,
            tenant_id=analysis.tenant_id,
            pages=ocr_pages,
            provider_name=self.provider.name,
            runtime_name="openvino",
            model_name="PP-OCRv3",
            config={"device": "CPU"},
            metadata={"document_type": analysis.document_type.value},
        )
