import pytest

from app.core.rag.contracts import (
    ChunkProvenance,
    ContentAuthority,
    RetrievalFilter,
    RetrievalPolicy,
)
from app.core.rag.errors import TenantContextMissingError
from app.services.rag.retrieval import HybridRetrievalEngine


@pytest.mark.asyncio
async def test_hybrid_retrieval_engine_index_and_retrieve():
    engine = HybridRetrievalEngine()

    prov = ChunkProvenance(
        source_uri="s3://school-assets/che101.pdf",
        page_number=12,
        section_title="Thermochemistry",
        authority=ContentAuthority.INSTITUTIONAL,
    )

    indexed = await engine.index_document(
        tenant_id="school_1",
        document_id="doc_chem_101",
        document_title="Chemistry Textbook Volume 1",
        text="Enthalpy is a thermodynamic quantity equivalent to the total heat content of a system. Hess's law states that total enthalpy change is independent of pathway.",
        course_code="CHE101",
        topic="Thermochemistry",
        provenance=prov,
    )
    assert indexed > 0

    # Query matching indexed document
    filter_obj = RetrievalFilter(tenant_id="school_1", course_code="CHE101")
    policy = RetrievalPolicy(top_k=3, dense_weight=0.6, sparse_weight=0.4)

    ctx = await engine.retrieve("What is Hess's law of enthalpy?", filter=filter_obj, policy=policy)

    assert ctx.tenant_id == "school_1"
    assert len(ctx.chunks) > 0
    assert ctx.total_candidates > 0
    assert ctx.chunks[0].chunk.document_title == "Chemistry Textbook Volume 1"
    assert ctx.chunks[0].hybrid_score > 0.0
    assert ctx.chunks[0].rank == 1

    citations = ctx.to_citation_summary()
    assert len(citations) > 0
    assert citations[0]["provenance"]["source_uri"] == "s3://school-assets/che101.pdf"


@pytest.mark.asyncio
async def test_hybrid_retrieval_engine_enforces_tenant_boundary():
    engine = HybridRetrievalEngine()

    # Index document in school_A
    await engine.index_document(
        tenant_id="school_A",
        document_id="doc_secret",
        document_title="School A Internal Exam Questions",
        text="Question 1: Derive the Navier-Stokes equations for incompressible fluid dynamics.",
        course_code="MTH401",
    )

    # Query from school_B
    filter_school_b = RetrievalFilter(tenant_id="school_B")
    ctx = await engine.retrieve("Navier-Stokes fluid dynamics", filter=filter_school_b)

    assert len(ctx.chunks) == 0
    assert ctx.total_candidates == 0


@pytest.mark.asyncio
async def test_hybrid_retrieval_missing_tenant_id_raises():
    engine = HybridRetrievalEngine()
    with pytest.raises(TenantContextMissingError):
        await engine.retrieve("test query", filter=RetrievalFilter(tenant_id=""))
