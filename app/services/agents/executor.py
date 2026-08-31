import json
import logging
import time

from app.core.agents.contracts import (
    AgentContext,
    AgentDefinition,
    AgentExecutionPolicy,
    AgentResult,
    AgentRun,
    AgentState,
)
from app.core.agents.errors import (
    AgentCapabilityDeniedError,
    AgentDisabledError,
    AgentNotFoundError,
    AgentPermissionDeniedError,
)
from app.core.agents.registry import AgentRegistry, get_agent_registry
from app.core.models.contracts import (
    ModelInvocationContext,
    ModelMessage,
    ModelRequest,
    ModelRole,
    ModelUsage,
)
from app.core.models.gateway import ModelGateway, get_model_gateway
from app.core.tools.contracts import ToolContext, ToolInvocation
from app.core.tools.registry import ToolRegistry, get_tool_registry
from app.services.tools.executor import ToolExecutor, get_tool_executor

logger = logging.getLogger(__name__)


class AgentExecutor:
    """
    Bounded Multi-Step Agent Orchestration Engine.
    Executes a structured ReAct loop while enforcing hard step, token, tool-call, runtime, and cost ceilings.
    """

    def __init__(
        self,
        model_gateway: ModelGateway | None = None,
        tool_registry: ToolRegistry | None = None,
        tool_executor: ToolExecutor | None = None,
        agent_registry: AgentRegistry | None = None,
    ):
        self.model_gateway = model_gateway or get_model_gateway()
        self.tool_registry = tool_registry or get_tool_registry()
        self.tool_executor = tool_executor or get_tool_executor()
        self.agent_registry = agent_registry or get_agent_registry()

    def _resolve_agent(self, agent_id: str, context: AgentContext) -> AgentDefinition:
        definition = self.agent_registry.get(agent_id)
        if not definition:
            raise AgentNotFoundError(agent_id)

        if not definition.is_enabled:
            raise AgentDisabledError(agent_id)

        # Check actor permissions
        if definition.required_permissions:
            actor_perms = set(context.permissions)
            missing = set(definition.required_permissions) - actor_perms
            if missing:
                raise AgentPermissionDeniedError(agent_id, list(missing))

        # Check actor capabilities
        if definition.required_capabilities:
            actor_caps = set(context.capabilities)
            missing = set(definition.required_capabilities) - actor_caps
            if missing:
                raise AgentCapabilityDeniedError(agent_id, list(missing))

        return definition

    async def run(
        self,
        agent_id: str,
        prompt: str,
        context: AgentContext,
        custom_policy: AgentExecutionPolicy | None = None,
    ) -> AgentResult:
        start_time = time.time()

        # 1. Resolve and authorize Agent
        try:
            definition = self._resolve_agent(agent_id, context)
        except Exception as exc:
            return AgentResult(
                run_id=context.agent_run_id,
                agent_id=agent_id,
                state=AgentState.FAILED,
                error_code=getattr(exc, "error_code", "AUTHORIZATION_FAILED"),
                error_message=str(exc),
                started_at=start_time,
                completed_at=time.time(),
                latency_ms=(time.time() - start_time) * 1000.0,
            )

        # 2. Initialize Agent Run tracker
        effective_policy = custom_policy or definition.execution_policy
        run = AgentRun(
            run_id=context.agent_run_id,
            agent_id=agent_id,
            agent_version=definition.version,
            tenant_id=context.tenant_id,
            execution_id=context.execution_id,
            task_id=context.task_id,
            correlation_id=context.correlation_id,
            state=AgentState.RUNNING,
            deadline=context.deadline,
        )

        # 3. Check pre-execution cancellation and deadline
        if context.is_cancelled():
            run.transition_to(AgentState.CANCELLED, error=context.cancellation_reason())
            return self._build_result(run, start_time)

        if context.is_expired():
            run.transition_to(AgentState.TIMED_OUT, error="Deadline expired before start")
            return self._build_result(run, start_time)

        # 4. Prepare Tooling Schema from ToolRegistry
        tool_ctx = ToolContext(
            tenant_id=context.tenant_id,
            execution_id=context.execution_id,
            task_id=context.task_id,
            agent_run_id=context.agent_run_id,
            correlation_id=context.correlation_id,
            actor_role=context.actor_role,
            permissions=context.permissions,
            capabilities=context.capabilities,
            deadline=context.deadline,
            _is_cancelled=context._is_cancelled,
        )

        available_tools = self.tool_registry.list_available(tool_ctx, policy=definition.tool_policy)
        tools_schema = [t.to_openai_tool() for t in available_tools] if available_tools else None

        # 5. Initialize Conversation History
        messages: list[ModelMessage] = [
            ModelMessage(role=ModelRole.SYSTEM, content=definition.system_prompt),
            ModelMessage(role=ModelRole.USER, content=prompt),
        ]

        final_output: str | None = None

        # 6. Bounded ReAct Execution Loop
        while run.is_active():
            # a. Cancellation check
            if context.is_cancelled():
                run.transition_to(AgentState.CANCELLED, error=context.cancellation_reason())
                break

            # b. Deadline & Runtime check
            now = time.time()
            if (
                context.is_expired()
                or (now - run.started_at) >= effective_policy.max_runtime_seconds
            ):
                run.transition_to(AgentState.TIMED_OUT, error="Execution runtime budget exceeded")
                break

            # c. Step budget check
            if run.step_count >= effective_policy.max_steps:
                run.transition_to(
                    AgentState.MAX_STEPS_EXCEEDED,
                    error=f"Max reasoning steps ({effective_policy.max_steps}) reached",
                )
                break

            # d. Model call budget check
            if run.model_call_count >= effective_policy.max_model_calls:
                run.transition_to(
                    AgentState.BUDGET_EXCEEDED,
                    error=f"Max model calls ({effective_policy.max_model_calls}) reached",
                )
                break

            # e. Token & Cost budget check
            if run.total_tokens >= effective_policy.max_tokens:
                run.transition_to(
                    AgentState.BUDGET_EXCEEDED,
                    error=f"Token budget ({effective_policy.max_tokens}) reached",
                )
                break

            if (
                effective_policy.max_cost_usd is not None
                and run.estimated_cost_usd >= effective_policy.max_cost_usd
            ):
                run.transition_to(
                    AgentState.BUDGET_EXCEEDED,
                    error=f"Cost budget (${effective_policy.max_cost_usd:.4f}) reached",
                )
                break

            # f. Dispatch Model Invocation via ModelGateway
            model_ctx = ModelInvocationContext(
                tenant_id=context.tenant_id,
                execution_id=context.execution_id,
                task_id=context.task_id,
                agent_run_id=context.agent_run_id,
                correlation_id=context.correlation_id,
            )

            model_request = ModelRequest(
                messages=messages,
                context=model_ctx,
                policy=definition.model_policy,
                tools=tools_schema,
            )

            try:
                model_response = await self.model_gateway.generate(model_request)
                run.record_model_call(model_response.usage)
                run.record_step()
                run.consecutive_failures = 0
            except Exception as model_err:
                logger.warning("Model invocation failed during agent step: %s", str(model_err))
                run.consecutive_failures += 1
                if run.consecutive_failures >= effective_policy.max_consecutive_failures:
                    run.transition_to(
                        AgentState.FAILED,
                        error=f"Consecutive model failure limit reached: {model_err!s}",
                    )
                    break
                else:
                    continue

            # g. Evaluate Decision: Final Answer vs Tool Calls
            tool_calls = model_response.tool_calls

            if not tool_calls:
                # Agent completed reasoning!
                final_output = model_response.content
                run.transition_to(AgentState.COMPLETED)
                break

            # h. Process Tool Invocations
            run.transition_to(AgentState.WAITING_FOR_TOOL)

            # Append assistant message with tool calls
            messages.append(
                ModelMessage(
                    role=ModelRole.ASSISTANT,
                    content=model_response.content or "",
                    tool_calls=tool_calls,
                )
            )

            for tc in tool_calls:
                # Check tool call budget
                if run.tool_call_count >= effective_policy.max_tool_calls:
                    run.transition_to(
                        AgentState.BUDGET_EXCEEDED,
                        error=f"Tool call budget ({effective_policy.max_tool_calls}) reached",
                    )
                    break

                fn_name = (
                    tc.get("function", {}).get("name")
                    if isinstance(tc.get("function"), dict)
                    else tc.get("name")
                )
                raw_args = (
                    tc.get("function", {}).get("arguments")
                    if isinstance(tc.get("function"), dict)
                    else tc.get("arguments", {})
                )
                call_id = tc.get("id", f"call_{run.tool_call_count}")

                if isinstance(raw_args, str):
                    try:
                        args_dict = json.loads(raw_args)
                    except Exception:
                        args_dict = {}
                elif isinstance(raw_args, dict):
                    args_dict = raw_args
                else:
                    args_dict = {}

                tool_inv = ToolInvocation(
                    tool_name=str(fn_name),
                    arguments=args_dict,
                    context=tool_ctx,
                )

                tool_result = await self.tool_executor.execute(
                    tool_inv, policy=definition.tool_policy
                )
                run.record_tool_call()

                # Format tool observation message
                if tool_result.success:
                    obs_content = (
                        json.dumps(tool_result.data)
                        if isinstance(tool_result.data, (dict, list))
                        else str(tool_result.data)
                    )
                else:
                    obs_content = f"Error ({tool_result.error_code}): {tool_result.error_message}"

                messages.append(
                    ModelMessage(
                        role=ModelRole.TOOL,
                        content=obs_content,
                        name=str(fn_name),
                        tool_call_id=call_id,
                    )
                )

            if run.state == AgentState.WAITING_FOR_TOOL:
                run.transition_to(AgentState.RUNNING)

        # 7. Final Result Construction
        return self._build_result(run, start_time, output=final_output)

    def _build_result(
        self, run: AgentRun, start_time: float, output: str | None = None
    ) -> AgentResult:
        latency_ms = (time.time() - start_time) * 1000.0
        return AgentResult(
            run_id=run.run_id,
            agent_id=run.agent_id,
            state=run.state,
            output=output,
            step_count=run.step_count,
            model_call_count=run.model_call_count,
            tool_call_count=run.tool_call_count,
            usage=ModelUsage(
                prompt_tokens=run.input_tokens,
                completion_tokens=run.output_tokens,
                total_tokens=run.total_tokens,
                estimated_cost_usd=run.estimated_cost_usd,
            ),
            estimated_cost_usd=run.estimated_cost_usd,
            started_at=run.started_at,
            completed_at=run.completed_at or time.time(),
            latency_ms=round(latency_ms, 2),
            error_code=run.state.value if run.state != AgentState.COMPLETED else None,
            error_message=run.last_error,
        )


_default_executor: AgentExecutor | None = None


def get_agent_executor() -> AgentExecutor:
    global _default_executor
    if _default_executor is None:
        _default_executor = AgentExecutor()
    return _default_executor


def set_agent_executor(executor: AgentExecutor) -> None:
    global _default_executor
    _default_executor = executor
