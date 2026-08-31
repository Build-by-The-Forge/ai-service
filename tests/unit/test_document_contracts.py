import pytest

from app.core.documents import (
    Document,
    DocumentMetadata,
    DocumentPage,
    DocumentSource,
    DocumentStatus,
    DocumentTenantMissingError,
    DocumentType,
    InvalidBoundingBoxError,
    OCRBlock,
    OCRBlockType,
    OCRBoundingBox,
    OCRConfig,
    OCRDocument,
    OCRPage,
    OCRPort,
    PageType,
)
from app.core.rag.contracts import ContentAuthority


def test_bounding_box_geometry_and_validation():
    # Valid box
    bbox = OCRBoundingBox(x_min=0.1, y_min=0.2, x_max=0.5, y_max=0.6)
    assert abs(bbox.width - 0.4) < 1e-4
    assert abs(bbox.height - 0.4) < 1e-4
    assert abs(bbox.area - 0.16) < 1e-4

    d = bbox.to_dict()
    assert d["x_min"] == 0.1
    assert d["x_max"] == 0.5

    # Inverted X coordinates should fail
    with pytest.raises(InvalidBoundingBoxError):
        OCRBoundingBox(x_min=0.8, y_min=0.2, x_max=0.3, y_max=0.6)

    # Inverted Y coordinates should fail
    with pytest.raises(InvalidBoundingBoxError):
        OCRBoundingBox(x_min=0.1, y_min=0.9, x_max=0.5, y_max=0.4)


def test_ocr_block_creation_and_clamping():
    bbox = OCRBoundingBox(x_min=10.0, y_min=20.0, x_max=100.0, y_max=50.0)
    block = OCRBlock.create(
        page_number=1,
        text="  CHAPTER 1: INTRODUCTION  ",
        confidence=1.5,  # Should clamp to 1.0
        bounding_box=bbox,
        block_type=OCRBlockType.HEADING,
        reading_order=1,
        metadata={"font_size": 18},
    )

    assert block.block_id.startswith("blk_")
    assert block.text == "CHAPTER 1: INTRODUCTION"
    assert block.confidence == 1.0
    assert block.block_type == OCRBlockType.HEADING
    assert block.metadata["font_size"] == 18


def test_document_creation_and_tenant_invariant():
    meta = DocumentMetadata(
        filename="lecture_1.pdf",
        mime_type="application/pdf",
        byte_size=102400,
        course_code="CSC201",
        authority=ContentAuthority.INSTITUTIONAL,
        storage_uri="s3://build-assets/lecture_1.pdf",
        content_sha256="abc123sha256hash",
    )

    # Valid creation
    doc = Document.create(
        tenant_id="school_unilag",
        title="CSC201 Lecture Notes",
        metadata=meta,
        doc_type=DocumentType.PDF,
        source=DocumentSource.UPLOAD,
    )

    assert doc.document_id.startswith("doc_")
    assert doc.tenant_id == "school_unilag"
    assert doc.status == DocumentStatus.PENDING
    assert doc.metadata.course_code == "CSC201"

    # Missing tenant_id must fail
    with pytest.raises(DocumentTenantMissingError):
        Document(
            document_id="doc_invalid",
            tenant_id="",
            title="No Tenant Doc",
            doc_type=DocumentType.PDF,
            source=DocumentSource.UPLOAD,
            status=DocumentStatus.PENDING,
            metadata=meta,
        )


def test_document_provenance_generation_for_rag():
    meta = DocumentMetadata(
        filename="physics_handbook.pdf",
        mime_type="application/pdf",
        byte_size=204800,
        course_code="PHY101",
        authority=ContentAuthority.APPROVED_REFERENCE,
        storage_uri="s3://build-assets/physics_handbook.pdf",
        content_sha256="def456sha256hash",
    )
    doc = Document.create(
        tenant_id="school_futa",
        title="Physics Reference Handbook",
        metadata=meta,
    )

    prov = doc.to_chunk_provenance(page_number=14, section_title="Thermodynamics Laws")
    assert prov.source_uri == "s3://build-assets/physics_handbook.pdf"
    assert prov.page_number == 14
    assert prov.section_title == "Thermodynamics Laws"
    assert prov.authority == ContentAuthority.APPROVED_REFERENCE
    assert prov.content_sha256 == "def456sha256hash"


def test_ocr_document_aggregation():
    bbox1 = OCRBoundingBox(x_min=0, y_min=0, x_max=100, y_max=30)
    bbox2 = OCRBoundingBox(x_min=0, y_min=40, x_max=100, y_max=80)

    b1 = OCRBlock.create(1, "Title of Page 1", 0.95, bbox1, OCRBlockType.HEADING, 1)
    b2 = OCRBlock.create(1, "First paragraph content.", 0.90, bbox2, OCRBlockType.PARAGRAPH, 2)

    page1 = OCRPage(
        page_number=1,
        blocks=[b1, b2],
        confidence=0.925,
        full_text="Title of Page 1\nFirst paragraph content.",
    )

    ocr_doc = OCRDocument(
        document_id="doc_100",
        tenant_id="school_oau",
        pages=[page1],
        overall_confidence=0.925,
        total_blocks=2,
        full_text=page1.full_text,
    )

    assert ocr_doc.document_id == "doc_100"
    assert ocr_doc.tenant_id == "school_oau"
    assert ocr_doc.total_blocks == 2
    assert len(ocr_doc.pages) == 1
    assert "Title of Page 1" in ocr_doc.full_text


def test_ocr_config_defaults_and_overrides():
    default_config = OCRConfig()
    assert default_config.device == "CPU"
    assert default_config.detection_input_size == 960
    assert default_config.recognition_batch_size == 8
    assert default_config.max_pages == 100

    custom_config = OCRConfig(
        detection_input_size=1280,
        recognition_batch_size=16,
        max_pages=250,
        device="GPU",
    )
    assert custom_config.detection_input_size == 1280
    assert custom_config.recognition_batch_size == 16
    assert custom_config.max_pages == 250
    assert custom_config.device == "GPU"


@pytest.mark.asyncio
async def test_ocr_port_mock_subclass():
    class TestMockOCRAdapter(OCRPort):
        async def extract_page(self, page: DocumentPage, config=None) -> OCRPage:
            bbox = OCRBoundingBox(x_min=0, y_min=0, x_max=10, y_max=10)
            block = OCRBlock.create(page.page_number, "Mock page text", 0.99, bbox)
            return OCRPage(
                page_number=page.page_number,
                blocks=[block],
                confidence=0.99,
                full_text="Mock page text",
            )

        async def extract_document(self, document: Document, config=None) -> OCRDocument:
            pages = []
            for p in document.pages:
                pages.append(await self.extract_page(p, config))
            return OCRDocument(
                document_id=document.document_id,
                tenant_id=document.tenant_id,
                pages=pages,
                overall_confidence=0.99,
                total_blocks=len(pages),
                full_text="\n".join(p.full_text for p in pages),
            )

        async def health_check(self):
            return {"status": "HEALTHY", "engine": "MockOCR"}

    adapter = TestMockOCRAdapter()
    health = await adapter.health_check()
    assert health["status"] == "HEALTHY"

    meta = DocumentMetadata(filename="doc.pdf", mime_type="application/pdf", byte_size=5000)
    doc = Document.create("school_1", "Test Doc", meta)
    doc.pages = [DocumentPage(page_number=1, page_type=PageType.SCANNED_IMAGE)]

    ocr_res = await adapter.extract_document(doc)
    assert ocr_res.tenant_id == "school_1"
    assert len(ocr_res.pages) == 1
    assert ocr_res.pages[0].blocks[0].text == "Mock page text"
