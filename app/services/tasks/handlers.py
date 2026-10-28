import abc
import asyncio
import time
from typing import Any

from app.core.rag.errors import (
    EmbeddingGenerationError,
    StaleIndexWriteError,
    VectorStoreError,
)
from app.services.ingestion.asset_fetcher import AssetFetcher
from app.services.ingestion.chunking import chunk_document
from app.services.ingestion.extraction import DocumentExtractor
from app.services.rag.embedder import EmbeddingPort
from app.services.rag.onnx_embedder import OnnxMiniLMEmbedder
from app.services.rag.pg_index import PgChunkIndex
from app.services.tasks.ingestion_outcomes import (
    PermanentIngestionError,
    PipelineUnavailableError,
    TransientIngestionError,
)
from app.core.execution_context import AIExecutionContext


class TaskHandler(abc.ABC):
    """
    Abstract interface for executing specific task types (e.g. content.ingestion, agent.run).
    """

    @abc.abstractmethod
    async def handle(self, ctx: AIExecutionContext, payload: dict[str, Any]) -> dict[str, Any]:
        """
        Executes the business logic for the task in the given execution context.
        Returns a result dictionary on success or raises an exception on failure.
        """


class ContentIngestionTaskHandler(TaskHandler):
    """
    Content ingestion: asset retrieval, extraction, chunking, embedding and
    durable indexing.

    1. Retrieval (P3-1): the dispatched asset is fetched from object storage
       and verified (managed reference of the verified tenant, SHA-256, size,
       type) before anything else happens.
    2. Extraction (P3-2): each page is read from its text layer or OCR'd
       (DocumentExtractor), and the "extracted" stage is reported through
       ctx.stage_reporter with its evidence.
    3. Chunking (P3-3): page-aware chunks with deterministic ids, and the
       "chunked" stage reported.
    4. Embedding and indexing (P3-4): every chunk is embedded locally, then
       the asset's chunks replace its previous ones in the durable index in
       one transaction. "indexed" is reported only after that commit, so the
       job can only become ready once its chunks are durably searchable.
       An embedding or database outage is transient (nothing partial is
       indexed); a missing embedding configuration is
       INGESTION_PIPELINE_UNAVAILABLE. Each failure carries the evidence of
       the stages before it.
    """

    _REQUIRED = ("asset_id", "storage_uri", "content_sha256", "mime_type")

    def __init__(
        self,
        fetcher: AssetFetcher | None = None,
        extractor: DocumentExtractor | None = None,
        embedder: EmbeddingPort | None = None,
        index: PgChunkIndex | None = None,
    ) -> None:
        self._fetcher = fetcher
        self._extractor = extractor
        self._embedder = embedder
        self._index = index

    def _text_embedder(self, evidence: dict[str, Any]) -> EmbeddingPort:
        # Built on first use and reused: the model inside it loads once.
        if self._embedder is None:
            try:
                self._embedder = OnnxMiniLMEmbedder.from_settings()
            except EmbeddingGenerationError as exc:
                raise PipelineUnavailableError(exc.message, details=evidence) from exc
        return self._embedder

    def _chunk_index(self) -> PgChunkIndex:
        if self._index is None:
            self._index = PgChunkIndex()
        return self._index

    def _document_extractor(self) -> DocumentExtractor:
        # Built on first use and reused: the OCR model inside it loads once.
        if self._extractor is None:
            self._extractor = DocumentExtractor.from_settings()
        return self._extractor

    def _asset_fetcher(self) -> AssetFetcher:
        # Built from settings on first use, then reused (one S3 client).
        if self._fetcher is None:
            self._fetcher = AssetFetcher.from_settings()
        return self._fetcher

    async def handle(self, ctx: AIExecutionContext, payload: dict[str, Any]) -> dict[str, Any]:
        if ctx.is_cancelled():
            raise asyncio.CancelledError(f"Task cancelled: {ctx.cancellation_reason}")
        missing = [name for name in self._REQUIRED if not payload.get(name)]
        if missing:
            raise PermanentIngestionError(
                "INVALID_INGESTION_PAYLOAD", f"the dispatch lacks {', '.join(missing)}"
            )

        started = time.monotonic()
        # The tenant is the verified context's, never the payload's.
        async with self._asset_fetcher().fetch(
            tenant_id=ctx.tenant_id,
            asset_id=payload["asset_id"],
            storage_uri=payload["storage_uri"],
            content_sha256=payload["content_sha256"],
            mime_type=payload["mime_type"],
            file_size_bytes=payload.get("file_size_bytes"),
        ) as asset:
            evidence: dict[str, Any] = {
                "fetch": {
                    "verified": True,
                    "size_bytes": asset.size_bytes,
                    "sha256": asset.sha256,
                    "mime_type": asset.mime_type,
                    "seconds": round(time.monotonic() - started, 3),
                }
            }
            if ctx.is_cancelled():
                raise asyncio.CancelledError(f"Task cancelled: {ctx.cancellation_reason}")
            document = await self._document_extractor().extract(asset.path, asset.mime_type)
        metrics = document.stage_metrics()
        if ctx.stage_reporter is not None:
            await ctx.stage_reporter.report("extracted", metrics)
        evidence["extraction"] = metrics
        if ctx.is_cancelled():
            raise asyncio.CancelledError(f"Task cancelled: {ctx.cancellation_reason}")
        # Chunk ids come from the verified tenant, the asset and the run, so
        # re-indexing a run replaces its chunks.
        run_number = int(payload.get("run_number") or 1)
        chunks = chunk_document(
            document.pages,
            tenant_id=ctx.tenant_id,
            asset_id=payload["asset_id"],
            run_number=run_number,
        )
        chunking = {
            "chunks": len(chunks),
            "characters": sum(len(chunk.text) for chunk in chunks),
            "pages_covered": len({p for c in chunks for p in range(c.page_start, c.page_end + 1)}),
        }
        if ctx.stage_reporter is not None:
            await ctx.stage_reporter.report("chunked", chunking)
        evidence["chunking"] = chunking
        if ctx.is_cancelled():
            raise asyncio.CancelledError(f"Task cancelled: {ctx.cancellation_reason}")

        started = time.monotonic()
        embedder = self._text_embedder(evidence)
        model = getattr(embedder, "model_id", type(embedder).__name__)
        try:
            vectors = await embedder.embed_batch([chunk.text for chunk in chunks])
        except EmbeddingGenerationError as exc:
            raise TransientIngestionError("EMBEDDING_FAILED", exc.message, evidence) from exc
        if ctx.is_cancelled():
            raise asyncio.CancelledError(f"Task cancelled: {ctx.cancellation_reason}")
        try:
            indexed = await self._chunk_index().replace_asset(
                tenant_id=ctx.tenant_id,
                asset_id=payload["asset_id"],
                run_number=run_number,
                attempt=int(payload.get("attempt_count") or 1),
                chunks=chunks,
                embeddings=vectors,
                embedding_model=model,
            )
        except StaleIndexWriteError as exc:
            # A newer attempt holds the asset's fence: this one must never
            # write, and repeating it cannot help (P3-6 §7.5).
            raise PermanentIngestionError("STALE_INDEX_WRITE", exc.message, evidence) from exc
        except VectorStoreError as exc:
            raise TransientIngestionError("INDEX_WRITE_FAILED", exc.message, evidence) from exc
        indexing = {
            "indexed_chunks": indexed,
            "index": "document_chunks",
            "embedding_model": model,
            "dimension": embedder.dimension,
            "seconds": round(time.monotonic() - started, 3),
        }
        # Committed: only now may the job advance towards ready.
        if ctx.stage_reporter is not None:
            await ctx.stage_reporter.report("indexed", indexing)
        evidence["indexing"] = indexing
        return {"chunk_summary": evidence}


class AgentExecutionTaskHandler(TaskHandler):
    """
    Task handler bridging background task worker executions to bounded agent orchestration.
    """

    def __init__(self, agent_executor: Any | None = None):
        self._executor = agent_executor

    async def handle(self, ctx: AIExecutionContext, payload: dict[str, Any]) -> dict[str, Any]:
        from app.core.agents.contracts import AgentContext
        from app.services.agents.executor import get_agent_executor

        if ctx.is_cancelled():
            raise asyncio.CancelledError(f"Task cancelled: {ctx.cancellation_reason}")

        agent_id = payload.get("agent_id", "study.assistant")
        prompt = payload.get("prompt", "")

        agent_ctx = AgentContext(
            tenant_id=ctx.tenant_id,
            execution_id=ctx.execution_id,
            task_id=ctx.execution_id,
            correlation_id=ctx.correlation_id,
            deadline=ctx.deadline,
            permissions=ctx.get_metadata(
                "permissions", ["curriculum:read", "content:read", "assessment:generate"]
            ),
            capabilities=ctx.get_metadata(
                "capabilities", ["curriculum_intelligence", "content_search", "cbt_generation"]
            ),
            _is_cancelled=ctx.is_cancelled(),
            _cancellation_reason=ctx.cancellation_reason,
        )

        executor = self._executor or get_agent_executor()
        result = await executor.run(agent_id=agent_id, prompt=prompt, context=agent_ctx)

        return {
            "status": result.state.value,
            "run_id": result.run_id,
            "agent_id": result.agent_id,
            "output": result.output,
            "step_count": result.step_count,
            "model_call_count": result.model_call_count,
            "tool_call_count": result.tool_call_count,
            "usage": {
                "prompt_tokens": result.usage.prompt_tokens,
                "completion_tokens": result.usage.completion_tokens,
                "total_tokens": result.usage.total_tokens,
                "estimated_cost_usd": result.estimated_cost_usd,
            },
            "error_code": result.error_code,
            "error_message": result.error_message,
        }


class TaskHandlerRegistry:
    """
    Registry mapping task_type string to TaskHandler implementations.
    """

    def __init__(self):
        self._handlers: dict[str, TaskHandler] = {}
        # Register default handlers
        self.register("content.ingestion", ContentIngestionTaskHandler())
        self.register("agent.execution", AgentExecutionTaskHandler())

    def register(self, task_type: str, handler: TaskHandler) -> None:
        self._handlers[task_type] = handler

    def get(self, task_type: str) -> TaskHandler | None:
        return self._handlers.get(task_type)


_default_registry: TaskHandlerRegistry | None = None


def get_task_handler_registry() -> TaskHandlerRegistry:
    global _default_registry
    if _default_registry is None:
        _default_registry = TaskHandlerRegistry()
    return _default_registry
