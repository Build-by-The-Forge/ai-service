"""The in-memory retrieval engine, for tests only: a deterministic hash
embedder, an in-memory vector store and an in-memory BM25 index. The app's
default engine is the durable one (retrieval.durable_retrieval_engine)."""

from app.services.rag.embedder import MockEmbeddingAdapter
from app.services.rag.lexical import BM25Index
from app.services.rag.retrieval import HybridRetrievalEngine
from app.services.rag.vector_store import InMemoryVectorStore


def in_memory_engine() -> HybridRetrievalEngine:
    return HybridRetrievalEngine(
        embedder=MockEmbeddingAdapter(),
        vector_store=InMemoryVectorStore(),
        lexical_index=BM25Index(),
    )
