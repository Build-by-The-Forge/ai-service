class AgentError(Exception):
    """Base exception for all agent runtime operations."""

    def __init__(self, message: str, agent_id: str | None = None, error_code: str = "AGENT_ERROR"):
        super().__init__(message)
        self.message = message
        self.agent_id = agent_id
        self.error_code = error_code


class AgentNotFoundError(AgentError):
    """Raised when an agent definition is not found in the registry."""

    def __init__(self, agent_id: str):
        super().__init__(
            f"Agent '{agent_id}' not found in registry",
            agent_id=agent_id,
            error_code="AGENT_NOT_FOUND",
        )


class AgentDisabledError(AgentError):
    """Raised when an agent definition is disabled."""

    def __init__(self, agent_id: str):
        super().__init__(
            f"Agent '{agent_id}' is disabled", agent_id=agent_id, error_code="AGENT_DISABLED"
        )


class AgentPermissionDeniedError(AgentError):
    """Raised when an actor lacks the required permissions for an agent."""

    def __init__(self, agent_id: str, missing_permissions: list[str]):
        super().__init__(
            f"Permission denied for agent '{agent_id}'. Missing: {missing_permissions}",
            agent_id=agent_id,
            error_code="PERMISSION_DENIED",
        )
        self.missing_permissions = missing_permissions


class AgentCapabilityDeniedError(AgentError):
    """Raised when an actor lacks the required capabilities for an agent."""

    def __init__(self, agent_id: str, missing_capabilities: list[str]):
        super().__init__(
            f"Capability denied for agent '{agent_id}'. Missing: {missing_capabilities}",
            agent_id=agent_id,
            error_code="CAPABILITY_DENIED",
        )
        self.missing_capabilities = missing_capabilities


class AgentBudgetExceededError(AgentError):
    """Raised when an agent run exhausts token, cost, model call, tool call, or runtime budgets."""

    def __init__(
        self,
        message: str,
        budget_type: str,
        limit: float,
        current: float,
        agent_id: str | None = None,
    ):
        super().__init__(message, agent_id=agent_id, error_code="AGENT_BUDGET_EXCEEDED")
        self.budget_type = budget_type
        self.limit = limit
        self.current = current


class AgentMaxStepsExceededError(AgentError):
    """Raised when an agent run exceeds its maximum allowed reasoning steps."""

    def __init__(self, agent_id: str, max_steps: int):
        super().__init__(
            f"Agent '{agent_id}' exceeded max allowed reasoning steps ({max_steps})",
            agent_id=agent_id,
            error_code="MAX_STEPS_EXCEEDED",
        )
        self.max_steps = max_steps


class AgentTimeoutError(AgentError):
    """Raised when an agent run exceeds its deadline or execution runtime limit."""

    def __init__(self, agent_id: str, timeout_seconds: float):
        super().__init__(
            f"Agent '{agent_id}' timed out after {timeout_seconds}s",
            agent_id=agent_id,
            error_code="AGENT_TIMEOUT",
        )
        self.timeout_seconds = timeout_seconds


class AgentCancelledError(AgentError):
    """Raised when an agent run is cancelled via cooperative cancellation token."""

    def __init__(self, agent_id: str, reason: str = "Execution cancelled"):
        super().__init__(
            f"Agent '{agent_id}' run cancelled: {reason}",
            agent_id=agent_id,
            error_code="AGENT_CANCELLED",
        )
        self.reason = reason


class AgentExecutionError(AgentError):
    """Raised when an unrecoverable runtime exception occurs during agent execution."""

    def __init__(self, agent_id: str, original_error: str):
        super().__init__(
            f"Agent '{agent_id}' execution failed: {original_error}",
            agent_id=agent_id,
            error_code="EXECUTION_FAILED",
        )
        self.original_error = original_error
