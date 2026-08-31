import asyncio
import logging
import time
from typing import Optional

from app.core.tools.budget import ToolBudgetTracker
from app.core.tools.contracts import (
    ToolExecutionPolicy,
    ToolInvocation,
    ToolResult,
    ToolSideEffect,
)
from app.core.tools.errors import ToolBudgetExceededError, ToolError
from app.core.tools.registry import ToolRegistry, get_tool_registry
from app.core.tools.validator import ToolValidator
from app.services.tools.builtins.content import (
    CONTENT_SEARCH_DEFINITION,
    content_search_handler,
)
from app.services.tools.builtins.curriculum import (
    CURRICULUM_LOOKUP_DEFINITION,
    curriculum_lookup_handler,
)
from app.services.tools.builtins.question import (
    DRAFT_QUESTION_DEFINITION,
    draft_question_handler,
)

logger = logging.getLogger(__name__)


class ToolExecutor:
    """
    Sandboxed Tool Execution Engine.
    Enforces authorization, tenant isolation, argument schemas, timeouts, byte limits, and execution budgets.
    """

    def __init__(
        self,
        registry: Optional[ToolRegistry] = None,
        budget_tracker: Optional[ToolBudgetTracker] = None,
        register_builtins: bool = True,
    ):
        self.registry = registry or get_tool_registry()
        self.budget_tracker = budget_tracker or ToolBudgetTracker()

        if register_builtins:
            self._register_default_tools()

    def _register_default_tools(self) -> None:
        self.registry.register(CURRICULUM_LOOKUP_DEFINITION, curriculum_lookup_handler)
        self.registry.register(CONTENT_SEARCH_DEFINITION, content_search_handler)
        self.registry.register(DRAFT_QUESTION_DEFINITION, draft_question_handler)

    async def execute(
        self,
        invocation: ToolInvocation,
        policy: Optional[ToolExecutionPolicy] = None,
    ) -> ToolResult:
        start_time = time.time()
        tool_name = invocation.tool_name
        scope_key = invocation.context.execution_id or invocation.context.tenant_id

        # 1. Resolve Tool from Registry
        entry = self.registry.get(tool_name)
        if not entry:
            logger.warning("Tool execution rejected: '%s' not found in registry", tool_name)
            return ToolResult(
                invocation_id=invocation.invocation_id,
                tool_name=tool_name,
                success=False,
                error_code="TOOL_NOT_FOUND",
                error_message=f"Tool '{tool_name}' is not registered",
                latency_ms=(time.time() - start_time) * 1000.0,
            )

        definition, handler = entry

        # 2. Check Enabled
        if not definition.is_enabled:
            return ToolResult(
                invocation_id=invocation.invocation_id,
                tool_name=tool_name,
                success=False,
                error_code="TOOL_DISABLED",
                error_message=f"Tool '{tool_name}' is currently disabled",
                latency_ms=(time.time() - start_time) * 1000.0,
            )

        # 3. Check Permissions & Capabilities
        if definition.required_permissions:
            actor_perms = set(invocation.context.permissions)
            missing_perms = set(definition.required_permissions) - actor_perms
            if missing_perms:
                logger.warning(
                    "Tool execution rejected: actor lacks permissions %s for tool %s",
                    missing_perms,
                    tool_name,
                )
                return ToolResult(
                    invocation_id=invocation.invocation_id,
                    tool_name=tool_name,
                    success=False,
                    error_code="PERMISSION_DENIED",
                    error_message=f"Missing required permissions: {list(missing_perms)}",
                    latency_ms=(time.time() - start_time) * 1000.0,
                )

        if definition.required_capabilities:
            actor_caps = set(invocation.context.capabilities)
            missing_caps = set(definition.required_capabilities) - actor_caps
            if missing_caps:
                return ToolResult(
                    invocation_id=invocation.invocation_id,
                    tool_name=tool_name,
                    success=False,
                    error_code="CAPABILITY_DENIED",
                    error_message=f"Missing required capabilities: {list(missing_caps)}",
                    latency_ms=(time.time() - start_time) * 1000.0,
                )

        # 4. Check Policy Allowlist & Side-effect constraints
        effective_policy = policy or ToolExecutionPolicy()

        if (
            effective_policy.allowed_tools is not None
            and tool_name not in effective_policy.allowed_tools
        ):
            return ToolResult(
                invocation_id=invocation.invocation_id,
                tool_name=tool_name,
                success=False,
                error_code="TOOL_NOT_ALLOWED",
                error_message=f"Tool '{tool_name}' is not in policy allowed_tools",
                latency_ms=(time.time() - start_time) * 1000.0,
            )

        if (
            definition.side_effect == ToolSideEffect.PROPOSAL
            and not effective_policy.allow_proposals
        ):
            return ToolResult(
                invocation_id=invocation.invocation_id,
                tool_name=tool_name,
                success=False,
                error_code="PROPOSALS_DISALLOWED",
                error_message=f"Proposal tool '{tool_name}' is disallowed by policy",
                latency_ms=(time.time() - start_time) * 1000.0,
            )

        if (
            definition.side_effect == ToolSideEffect.MUTATING
            and not effective_policy.allow_mutating
        ):
            return ToolResult(
                invocation_id=invocation.invocation_id,
                tool_name=tool_name,
                success=False,
                error_code="MUTATION_DISALLOWED",
                error_message=f"Mutating tool '{tool_name}' is disallowed by policy",
                latency_ms=(time.time() - start_time) * 1000.0,
            )

        # 5. Validate Schema, Byte Size & Authoritative Tenant Boundary
        try:
            ToolValidator.validate_input(definition, invocation, effective_policy)
        except ToolError as val_err:
            return ToolResult(
                invocation_id=invocation.invocation_id,
                tool_name=tool_name,
                success=False,
                error_code=val_err.error_code,
                error_message=val_err.message,
                latency_ms=(time.time() - start_time) * 1000.0,
            )

        # 6. Check Execution Budgets
        try:
            self.budget_tracker.check_pre_invocation(scope_key, effective_policy)
        except ToolBudgetExceededError as budget_err:
            return ToolResult(
                invocation_id=invocation.invocation_id,
                tool_name=tool_name,
                success=False,
                error_code=budget_err.error_code,
                error_message=budget_err.message,
                latency_ms=(time.time() - start_time) * 1000.0,
            )

        # 7. Execute Handler with Timeout
        timeout = min(definition.timeout_seconds, effective_policy.max_runtime_seconds)

        try:
            raw_result = await asyncio.wait_for(
                handler(invocation.context, invocation.arguments),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            logger.warning("Tool '%s' execution timed out after %fs", tool_name, timeout)
            return ToolResult(
                invocation_id=invocation.invocation_id,
                tool_name=tool_name,
                success=False,
                error_code="TOOL_TIMEOUT",
                error_message=f"Tool timed out after {timeout}s",
                latency_ms=(time.time() - start_time) * 1000.0,
            )
        except asyncio.CancelledError:
            logger.info("Tool '%s' execution was cancelled", tool_name)
            return ToolResult(
                invocation_id=invocation.invocation_id,
                tool_name=tool_name,
                success=False,
                error_code="EXECUTION_CANCELLED",
                error_message="Tool execution cancelled",
                latency_ms=(time.time() - start_time) * 1000.0,
            )
        except Exception as exc:
            logger.error(
                "Tool '%s' unhandled runtime exception: %s", tool_name, str(exc), exc_info=True
            )
            return ToolResult(
                invocation_id=invocation.invocation_id,
                tool_name=tool_name,
                success=False,
                error_code="EXECUTION_FAILED",
                error_message=str(exc),
                latency_ms=(time.time() - start_time) * 1000.0,
            )

        # 8. Validate Output Payload Size Limit
        try:
            output_bytes = ToolValidator.validate_output(tool_name, raw_result, effective_policy)
        except ToolError as out_err:
            return ToolResult(
                invocation_id=invocation.invocation_id,
                tool_name=tool_name,
                success=False,
                error_code=out_err.error_code,
                error_message=out_err.message,
                latency_ms=(time.time() - start_time) * 1000.0,
            )

        latency_seconds = time.time() - start_time
        latency_ms = latency_seconds * 1000.0

        # 9. Record Budget Telemetry
        self.budget_tracker.record_invocation(scope_key, latency_seconds)

        return ToolResult(
            invocation_id=invocation.invocation_id,
            tool_name=tool_name,
            success=True,
            data=raw_result,
            latency_ms=round(latency_ms, 2),
            output_bytes=output_bytes,
        )


_default_executor: Optional[ToolExecutor] = None


def get_tool_executor() -> ToolExecutor:
    global _default_executor
    if _default_executor is None:
        _default_executor = ToolExecutor()
    return _default_executor


def set_tool_executor(executor: ToolExecutor) -> None:
    global _default_executor
    _default_executor = executor
