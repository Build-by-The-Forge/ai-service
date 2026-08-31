from app.core.rag.contracts import (
    ChunkProvenance,
    ContentAuthority,
    DocumentChunk,
    RetrievedChunk,
    RetrievedContext,
)


def test_document_chunk_and_provenance_creation():
    prov = ChunkProvenance(
        source_uri="s3://build-assets/syllabus.pdf",
        page_number=3,
        section_title="Data Structures",
        authority=ContentAuthority.INSTITUTIONAL,
        content_sha256="abc123hash",
    )
    chunk = DocumentChunk.create(
        tenant_id="school_1",
        document_id="doc_1",
        document_title="CS Syllabus",
        content="Binary trees are hierarchical data structures.",
        course_code="CSC201",
        topic="Binary Trees",
        provenance=prov,
    )

    assert chunk.chunk_id.startswith("chk_")
    assert chunk.tenant_id == "school_1"
    assert chunk.course_code == "CSC201"
    assert chunk.provenance is not None
    assert chunk.provenance.page_number == 3

    citation = chunk.provenance.to_citation_dict()
    assert citation["source_uri"] == "s3://build-assets/syllabus.pdf"
    assert citation["authority"] == "INSTITUTIONAL"


def test_retrieved_context_citation_summary():
    chunk = DocumentChunk.create(
        tenant_id="school_1",
        document_id="doc_1",
        document_title="Algorithms Notes",
        content="Quicksort has an average time complexity of O(N log N).",
        course_code="CSC201",
    )
    retrieved_chunk = RetrievedChunk(
        chunk=chunk,
        dense_score=0.92,
        sparse_score=0.85,
        hybrid_score=0.89,
        rank=1,
    )
    ctx = RetrievedContext(
        query="What is the complexity of quicksort?",
        tenant_id="school_1",
        chunks=[retrieved_chunk],
        retrieval_method="hybrid_rrf",
        latency_ms=12.5,
        total_candidates=1,
    )

    citations = ctx.to_citation_summary()
    assert len(citations) == 1
    assert citations[0]["document_title"] == "Algorithms Notes"
    assert citations[0]["hybrid_score"] == 0.89
    assert citations[0]["rank"] == 1
