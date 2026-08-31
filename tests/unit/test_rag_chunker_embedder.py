import pytest

from app.services.rag.chunker import Chunker
from app.services.rag.embedder import (
    MockEmbeddingAdapter,
    cosine_similarity,
    l2_normalize,
)


def test_chunker_sliding_window_and_metadata():
    chunker = Chunker(chunk_size=10, chunk_overlap=3)
    text = "word1 word2 word3 word4 word5 word6 word7 word8 word9 word10 word11 word12 word13 word14 word15"

    chunks = chunker.chunk_document(
        tenant_id="school_1",
        document_id="doc_1",
        document_title="Test Doc",
        text=text,
        course_code="PHY101",
    )

    assert len(chunks) >= 2
    assert chunks[0].tenant_id == "school_1"
    assert chunks[0].course_code == "PHY101"
    assert chunks[0].metadata["chunk_index"] == 1
    assert chunks[0].metadata["total_chunks"] == len(chunks)


@pytest.mark.asyncio
async def test_mock_embedding_adapter_deterministic_and_normalized():
    adapter = MockEmbeddingAdapter(dimension=128)
    assert adapter.dimension == 128

    vec1 = await adapter.embed_text("Quantum mechanics and wave functions")
    vec2 = await adapter.embed_text("Quantum mechanics and wave functions")
    vec3 = await adapter.embed_text("Photosynthesis in green plants")

    # Determinism
    assert vec1 == vec2

    # L2 normalized
    norm = sum(x * x for x in vec1)
    assert abs(norm - 1.0) < 1e-4

    # Similarity checks
    sim_same = cosine_similarity(vec1, vec2)
    sim_diff = cosine_similarity(vec1, vec3)

    assert abs(sim_same - 1.0) < 1e-4
    assert sim_diff < 0.99


def test_vector_math_edge_cases():
    assert cosine_similarity([], []) == 0.0
    assert l2_normalize([0.0, 0.0, 0.0]) == [0.0, 0.0, 0.0]
