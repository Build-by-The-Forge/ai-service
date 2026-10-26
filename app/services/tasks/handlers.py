import abc
import asyncio
import time
from typing import Any

from app.services.ingestion.asset_fetcher import AssetFetcher
from app.services.tasks.ingestion_outcomes import PermanentIngestionError, PipelineUnavailableError
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

    Asset retrieval exists (P3-1 PR 5): the dispatched asset is fetched from
    object storage and verified (managed reference of the verified tenant,
    SHA-256, size, type) before anything else happens. Nothing downstream
    exists yet, so after a verified fetch the task still fails honestly with
    INGESTION_PIPELINE_UNAVAILABLE, carrying the fetch evidence; a fetch that
    fails reports its own typed failure. A real pipeline reports each stage
    through ctx.stage_reporter.
    """

    _REQUIRED = ("asset_id", "storage_uri", "content_sha256", "mime_type")

    def __init__(self, fetcher: AssetFetcher | None = None) -> None:
        self._fetcher = fetcher

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
            evidence = {
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
            # Extraction (P3-2) starts here, reading asset.path.
            raise PipelineUnavailableError(
                "The asset was fetched and verified; extraction, chunking, embedding and "
                "durable indexing are not available in this deployment yet.",
                details=evidence,
            )


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
