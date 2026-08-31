class RAGError(Exception):
    """Base exception for all RAG and retrieval operations."""

    def __init__(self, message: str, error_code: str = "RAG_ERROR", tenant_id: str | None = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.tenant_id = tenant_id


class TenantContextMissingError(RAGError):
    """Raised when a retrieval operation is attempted without an authoritative tenant context."""

    def __init__(self, message: str = "Retrieval operation requires an authoritative tenant_id"):
        super().__init__(message, error_code="TENANT_CONTEXT_MISSING")


class EmbeddingGenerationError(RAGError):
    """Raised when an embedding provider fails to generate vector embeddings."""

    def __init__(self, message: str, provider: str = "unknown"):
        super().__init__(
            f"Embedding generation failed on {provider}: {message}",
            error_code="EMBEDDING_GENERATION_FAILED",
        )
        self.provider = provider


class VectorStoreError(RAGError):
    """Raised when vector storage or similarity query encounters a database error."""

    def __init__(self, message: str, store_type: str = "unknown"):
        super().__init__(
            f"Vector store ({store_type}) error: {message}", error_code="VECTOR_STORE_ERROR"
        )
        self.store_type = store_type


class DocumentNotFoundError(RAGError):
    """Raised when a requested document ID or chunk ID does not exist."""

    def __init__(self, document_id: str, tenant_id: str | None = None):
        super().__init__(
            f"Document '{document_id}' not found",
            error_code="DOCUMENT_NOT_FOUND",
            tenant_id=tenant_id,
        )
        self.document_id = document_id


class RetrievalBudgetExceededError(RAGError):
    """Raised when a retrieval query exceeds maximum context token or candidate limits."""

    def __init__(self, message: str, limit: int, current: int):
        super().__init__(message, error_code="RETRIEVAL_BUDGET_EXCEEDED")
        self.limit = limit
        self.current = current
