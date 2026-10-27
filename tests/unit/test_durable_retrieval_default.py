"""The app's default retrieval engine is the durable one (P3-5): the local
embedder and the durable index, never the in-memory test engine."""

import pytest

from app.core.rag.errors import EmbeddingGenerationError
from app.services.rag.onnx_embedder import OnnxMiniLMEmbedder
from app.services.rag.pg_index import PgChunkIndex
from app.services.rag.retrieval import get_retrieval_engine, set_retrieval_engine


@pytest.fixture
def no_engine():
    set_retrieval_engine(None)  # undo the test default (tests/conftest.py)
    yield
    set_retrieval_engine(None)


def test_the_default_engine_is_the_durable_index_and_local_model(no_engine, monkeypatch, tmp_path):
    from app.core.config import settings

    monkeypatch.setattr(settings, "EMBEDDING_MODEL_DIR", str(tmp_path))  # loaded lazily

    engine = get_retrieval_engine()

    assert isinstance(engine.embedder, OnnxMiniLMEmbedder)
    assert isinstance(engine.vector_store, PgChunkIndex)
    # One table serves both sides: the same tenant-filtered index.
    assert engine.lexical_index is engine.vector_store
    assert get_retrieval_engine() is engine


def test_without_the_model_configured_there_is_no_silent_fallback(no_engine, monkeypatch):
    from app.core.config import settings

    monkeypatch.setattr(settings, "EMBEDDING_MODEL_DIR", None)
    with pytest.raises(EmbeddingGenerationError, match="EMBEDDING_MODEL_DIR is not set"):
        get_retrieval_engine()
