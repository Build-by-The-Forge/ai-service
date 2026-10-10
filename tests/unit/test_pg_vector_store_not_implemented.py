"""PgVectorStore must not report storage or search it did not perform
(Phase 2 audit F1). It is replaced by a real implementation in durable
indexing; until then every operation fails explicitly."""

import pytest

from app.core.rag.contracts import RetrievalFilter
from app.core.rag.errors import TenantContextMissingError, VectorStoreNotImplementedError
from app.services.rag.vector_store import PgVectorStore


@pytest.mark.parametrize("session_factory", [None, object()], ids=["unconfigured", "configured"])
@pytest.mark.parametrize(
    "operation,call",
    [
        ("add_chunks", lambda s: s.add_chunks([])),
        ("query_dense", lambda s: s.query_dense([0.1], RetrievalFilter(tenant_id="t1"))),
        ("delete_document", lambda s: s.delete_document("t1", "doc")),
        ("count", lambda s: s.count("t1")),
    ],
)
async def test_every_operation_fails_explicitly(session_factory, operation, call):
    with pytest.raises(VectorStoreNotImplementedError) as exc:
        await call(PgVectorStore(session_factory=session_factory))
    assert exc.value.error_code == "VECTOR_STORE_NOT_IMPLEMENTED"
    assert exc.value.operation == operation
    assert "pgvector" in str(exc.value)


async def test_a_query_without_a_tenant_is_still_refused_first():
    with pytest.raises(TenantContextMissingError):
        await PgVectorStore().query_dense([0.1], RetrievalFilter(tenant_id=""))
