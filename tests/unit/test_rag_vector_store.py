import pytest

from app.core.rag.contracts import DocumentChunk, RetrievalFilter
from app.core.rag.errors import TenantContextMissingError
from app.services.rag.embedder import MockEmbeddingAdapter
from app.services.rag.lexical import BM25Index
from app.services.rag.vector_store import InMemoryVectorStore


@pytest.mark.asyncio
async def test_in_memory_vector_store_dense_retrieval_and_tenant_isolation():
    store = InMemoryVectorStore()
    embedder = MockEmbeddingAdapter(dimension=64)

    vec_t1 = await embedder.embed_text("Operating systems memory management and paging")
    chunk_t1 = DocumentChunk.create(
        tenant_id="school_1",
        document_id="doc_os",
        document_title="OS Lecture 3",
        content="Paging divides memory into fixed size blocks.",
        course_code="CSC301",
        embedding=vec_t1,
    )

    vec_t2 = await embedder.embed_text("Operating systems memory management and paging")
    chunk_t2 = DocumentChunk.create(
        tenant_id="school_2",
        document_id="doc_os_2",
        document_title="Secret School 2 Notes",
        content="School 2 confidential paging notes.",
        course_code="CSC301",
        embedding=vec_t2,
    )

    await store.add_chunks([chunk_t1, chunk_t2])
    assert await store.count("school_1") == 1
    assert await store.count("school_2") == 1

    # Search for school_1 only
    query_vec = await embedder.embed_text("memory management paging")
    filter_t1 = RetrievalFilter(tenant_id="school_1", course_code="CSC301")
    results = await store.query_dense(query_vec, filter=filter_t1, top_k=5)

    assert len(results) == 1
    matched_chunk, score = results[0]
    assert matched_chunk.tenant_id == "school_1"
    assert matched_chunk.document_title == "OS Lecture 3"
    assert score > 0.5

    # Cross-tenant check: school_2 should never appear
    assert all(c.tenant_id == "school_1" for c, _ in results)


@pytest.mark.asyncio
async def test_bm25_lexical_index_sparse_keyword_matching():
    index = BM25Index()

    chunk1 = DocumentChunk.create(
        tenant_id="school_1",
        document_id="doc_bio",
        document_title="Biology 101",
        content="Photosynthesis occurs in chloroplasts using chlorophyll and sunlight.",
        course_code="BIO101",
    )
    chunk2 = DocumentChunk.create(
        tenant_id="school_1",
        document_id="doc_chem",
        document_title="Chemistry 101",
        content="Covalent bonds involve sharing electrons between atoms in molecules.",
        course_code="CHM101",
    )

    await index.index_chunks([chunk1, chunk2])

    filter_bio = RetrievalFilter(tenant_id="school_1")
    results = await index.query_sparse("chloroplasts chlorophyll", filter=filter_bio, top_k=5)

    assert len(results) == 1
    matched, score = results[0]
    assert matched.document_title == "Biology 101"
    assert score > 0.0


@pytest.mark.asyncio
async def test_vector_store_requires_tenant_id():
    store = InMemoryVectorStore()
    with pytest.raises(TenantContextMissingError):
        await store.query_dense([0.1, 0.2], filter=RetrievalFilter(tenant_id=""))
