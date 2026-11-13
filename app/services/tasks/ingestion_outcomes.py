"""How an ingestion task ends, and how it reports stages (M3-08).

Failures are classified deliberately, never from exception text:

* ``TransientIngestionError``: likely to succeed if retried (a storage or
  dependency outage). The worker retries it internally; if it is still
  failing, the backend is told the attempt may be retried.
* ``PermanentIngestionError``: repeating the same work cannot succeed
  (unsupported or irrecoverably invalid content). Not retried.
* ``PipelineUnavailableError``: a deployment or configuration problem: a
  required pipeline capability is missing. Not retried for this attempt;
  an authorized reprocess may succeed once the deployment changes.

``retryable`` is reported to the backend: whether the current attempt may be
retried automatically under present conditions. It never prevents a
separately authorized reprocess.

``StageReporter`` is how a pipeline reports real progress: extracted, then
chunked, then indexed, each with its evidence, in order, each delivered
before the next. A stage that cannot be delivered stops the pipeline. A job
can only complete after "indexed" has been reported.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

PIPELINE_UNAVAILABLE = "INGESTION_PIPELINE_UNAVAILABLE"

# Every code an ingestion failure can report to the backend (#54): a closed set,
# each taken from the failure's category (its class or the ingestion outcome),
# never from its text, so the backend can map each to a user-safe message.
# tests/unit/test_ingestion_failure_callbacks.py checks that every code raised
# in app/ is listed here.
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
        "INGESTION_INCOMPLETE",
        "INGESTION_PIPELINE_UNAVAILABLE",
        "INVALID_INGESTION_PAYLOAD",
        "STAGE_OUT_OF_ORDER",
        "UNREGISTERED_HANDLER",
        "AI_UNCLASSIFIED_ERROR",
        "AI_PROCESSING_FAILED",  # the callback's default when an event has no code
    }
)
STAGES = ("extracted", "chunked", "indexed")


class IngestionFailure(Exception):
    """Base class for classified ingestion failures."""

    retryable = False

    def __init__(self, code: str, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        # Evidence for the backend (job error_details): what was done before
        # the failure, e.g. a verified fetch. Data only, never secrets.
        self.details = dict(details or {})


class TransientIngestionError(IngestionFailure):
    retryable = True


class PermanentIngestionError(IngestionFailure):
    retryable = False


class PipelineUnavailableError(IngestionFailure):
    retryable = False

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(PIPELINE_UNAVAILABLE, message, details)


class StageReporter:
    """Reports pipeline stages in order through a delivery function.

    ``deliver(stage, metrics)`` returns (delivered, retryable). A failed
    delivery raises a classified failure, so the pipeline stops and no later
    stage, or completion, is ever reported without its predecessor.
    """

    def __init__(self, deliver: Callable[[str, dict[str, Any]], Awaitable[tuple[bool, bool]]]):
        self._deliver = deliver
        self.reached: list[str] = []

    @property
    def indexed(self) -> bool:
        return self.reached == list(STAGES)

    async def report(self, stage: str, metrics: dict[str, Any]) -> None:
        expected = STAGES[len(self.reached)] if len(self.reached) < len(STAGES) else None
        if stage != expected:
            raise PermanentIngestionError(
                "STAGE_OUT_OF_ORDER",
                f"stage {stage!r} reported, but the next stage is {expected!r}",
            )
        delivered, retryable = await self._deliver(stage, metrics)
        if not delivered:
            error = TransientIngestionError if retryable else PermanentIngestionError
            raise error("CALLBACK_DELIVERY_FAILED", f"stage {stage!r} could not be delivered")
        self.reached.append(stage)
