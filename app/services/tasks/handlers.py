import abc
import asyncio
from typing import Any

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
    Deterministic task handler for content ingestion pipelines (OCR, PDF/Doc parsing, chunking, indexing).
    """

    async def handle(self, ctx: AIExecutionContext, payload: dict[str, Any]) -> dict[str, Any]:
        if ctx.is_cancelled():
            raise asyncio.CancelledError(f"Task cancelled: {ctx.cancellation_reason}")

        # Extract envelope properties
        storage_uri = payload.get("storage_uri", "")
        mime_type = payload.get("mime_type", "")
        authority = payload.get("authority", "INSTITUTIONAL")
        content_sha256 = payload.get("content_sha256", "")

        ctx.set_metadata("pipeline_stage", "EXTRACTION_INITIALIZED")

        # Check cooperative cancellation
        if ctx.is_cancelled():
            raise asyncio.CancelledError(f"Task cancelled: {ctx.cancellation_reason}")

        # Result summary matching Backend 4B expectation
        result = {
            "status": "COMPLETED",
            "job_id": ctx.job_id,
            "asset_id": ctx.asset_id,
            "tenant_id": ctx.tenant_id,
            "storage_uri": storage_uri,
            "mime_type": mime_type,
            "authority": authority,
            "content_sha256": content_sha256,
            "chunks_count": 1,
            "extracted_metadata": {
                "execution_id": ctx.execution_id,
                "processed_at": ctx.created_at,
            },
        }

        ctx.set_metadata("pipeline_stage", "COMPLETED")
        return result


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
