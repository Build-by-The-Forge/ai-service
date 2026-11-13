import asyncio
import logging
import time

from app.core.safe_log import describe
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

# What the agent reads about a failure (#53): a fixed message per error code,
# never the exception's text, which can quote a passage, an argument or a
# provider's response. The code and the retry flag come from the failure's
# category (its class or typed code), never from its message.
FAILURE_MESSAGES: dict[str, str] = {
    "TOOL_EXECUTION_FAILED": "The tool could not complete this operation.",
    "TOOL_TIMEOUT": "The tool did not finish in time.",
    "EXECUTION_CANCELLED": "The tool's run was cancelled.",
    "TOOL_BUDGET_EXCEEDED": "The tool budget for this run is used up.",
    "INVALID_PARAMETERS": "The tool's arguments were not valid.",
    "TENANT_MISMATCH": "The tool can act only within the caller's institution.",
    "PERMISSION_DENIED": "The tool is not permitted here.",
    "TOOL_NOT_FOUND": "The tool is not available.",
    "TOOL_DISABLED": "The tool is not available.",
}
RETRYABLE_CODES = frozenset({"TOOL_TIMEOUT"})


def failure_message(code: str) -> str:
    return FAILURE_MESSAGES.get(code, FAILURE_MESSAGES["TOOL_EXECUTION_FAILED"])


class ToolExecutor:
    """
    Sandboxed Tool Execution Engine.
    Enforces authorization, tenant isolation, argument schemas, timeouts, byte limits, and execution budgets.
    """

    def __init__(
        self,
        registry: ToolRegistry | None = None,
        budget_tracker: ToolBudgetTracker | None = None,
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

    @staticmethod
    def _failed(invocation: ToolInvocation, code: str, start_time: float) -> ToolResult:
        """A failure the agent may read: the code, its fixed message and whether
        it is worth retrying, from the failure's category (#53)."""
        return ToolResult(
            invocation_id=invocation.invocation_id,
            tool_name=invocation.tool_name,
            success=False,
            error_code=code,
            error_message=failure_message(code),
            retryable=code in RETRYABLE_CODES,
            latency_ms=(time.time() - start_time) * 1000.0,
        )

    async def execute(
        self,
        invocation: ToolInvocation,
        policy: ToolExecutionPolicy | None = None,
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
            return self._failed(invocation, val_err.error_code, start_time)

        # 6. Check Execution Budgets
        try:
            self.budget_tracker.check_pre_invocation(scope_key, effective_policy)
        except ToolBudgetExceededError as budget_err:
            return self._failed(invocation, budget_err.error_code, start_time)

        # 7. Execute Handler with Timeout
        timeout = min(definition.timeout_seconds, effective_policy.max_runtime_seconds)

        try:
            raw_result = await asyncio.wait_for(
                handler(invocation.context, invocation.arguments),
                timeout=timeout,
            )
        except TimeoutError:
            logger.warning("Tool '%s' execution timed out after %fs", tool_name, timeout)
            return self._failed(invocation, "TOOL_TIMEOUT", start_time)
        except asyncio.CancelledError:
            logger.info("Tool '%s' execution was cancelled", tool_name)
            return self._failed(invocation, "EXECUTION_CANCELLED", start_time)
        except ToolError as tool_err:
            # A typed failure keeps its category's code; its text stays here.
            logger.warning("Tool '%s' failed: %s", tool_name, describe(tool_err))
            return self._failed(invocation, tool_err.error_code, start_time)
        except Exception as exc:
            # Type and place only: a tool handles passages, and its
            # exception's text may quote one.
            logger.error("Tool '%s' unhandled runtime exception: %s", tool_name, describe(exc))
            return self._failed(invocation, "TOOL_EXECUTION_FAILED", start_time)

        # 8. Validate Output Payload Size Limit
        try:
            output_bytes = ToolValidator.validate_output(tool_name, raw_result, effective_policy)
        except ToolError as out_err:
            return self._failed(invocation, out_err.error_code, start_time)

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


_default_executor: ToolExecutor | None = None


def get_tool_executor() -> ToolExecutor:
    global _default_executor
    if _default_executor is None:
        _default_executor = ToolExecutor()
    return _default_executor


def set_tool_executor(executor: ToolExecutor) -> None:
    global _default_executor
    _default_executor = executor
