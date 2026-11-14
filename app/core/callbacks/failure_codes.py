"""The failure codes an ingestion callback may carry (#54)."""

# Every code an ingestion failure can report to the backend (#54): a closed set,
# each taken from the failure's category (its class or the ingestion outcome),
# never from its text, so the backend can map each to a user-safe message.
# tests/unit/test_failure_code_set.py reads app/'s syntax tree, aliases and
# constants included, and checks that every code raised there is listed here;
# backend_callback_body sends no other code.
FAILURE_CODES = frozenset(
    {
        # Fetching the asset from object storage.
        "ASSET_FETCH_TIMEOUT",
        "ASSET_INTEGRITY_MISMATCH",
        "ASSET_NOT_FOUND",
        "ASSET_REFERENCE_REJECTED",
        "ASSET_TOO_LARGE",
        "ASSET_TOO_MANY_PAGES",
        "ASSET_TYPE_MISMATCH",
        "OBJECT_STORAGE_ACCESS_DENIED",
        "OBJECT_STORAGE_FAILURE",
        "OBJECT_STORAGE_MISCONFIGURED",
        "OBJECT_STORAGE_NOT_CONFIGURED",
        "OBJECT_STORAGE_UNAVAILABLE",
        # Extracting text.
        "EXTRACTION_EMPTY",
        "EXTRACTION_TIMEOUT",
        "IMAGE_UNREADABLE",
        "OCR_PAGE_FAILED",
        "PDF_ENCRYPTED",
        "PDF_UNREADABLE",
        # Embedding and indexing.
        "EMBEDDING_FAILED",
        "INDEX_WRITE_FAILED",
        "STALE_INDEX_WRITE",
        # The pipeline and the worker.
        "CALLBACK_DELIVERY_FAILED",  # a stage callback the backend didn't accept
        "INGESTION_INCOMPLETE",
        "INGESTION_PIPELINE_UNAVAILABLE",
        "INVALID_INGESTION_PAYLOAD",
        "STAGE_OUT_OF_ORDER",
        "UNREGISTERED_HANDLER",
        "AI_UNCLASSIFIED_ERROR",
        "AI_PROCESSING_FAILED",  # the callback's default when an event has no code
    }
)

# What the callback reports instead of a code outside the set: a defect to
# investigate, never retried.
UNLISTED_FAILURE_CODE = "AI_UNCLASSIFIED_ERROR"
