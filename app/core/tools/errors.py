class ToolError(Exception):
    """Base exception for all tool operations."""

    def __init__(self, message: str, tool_name: str | None = None, error_code: str = "TOOL_ERROR"):
        super().__init__(message)
        self.message = message
        self.tool_name = tool_name
        self.error_code = error_code


class ToolNotFoundError(ToolError):
    """Raised when a requested tool is not registered in the ToolRegistry."""

    def __init__(self, tool_name: str):
        super().__init__(
            f"Tool '{tool_name}' not found in registry",
            tool_name=tool_name,
            error_code="TOOL_NOT_FOUND",
        )


class ToolDisabledError(ToolError):
    """Raised when a requested tool is currently disabled in the platform."""

    def __init__(self, tool_name: str):
        super().__init__(
            f"Tool '{tool_name}' is disabled", tool_name=tool_name, error_code="TOOL_DISABLED"
        )


class ToolPermissionDeniedError(ToolError):
    """Raised when an actor lacks the required permissions or capabilities for a tool."""

    def __init__(self, tool_name: str, missing_permissions: list[str]):
        super().__init__(
            f"Permission denied for tool '{tool_name}'. Missing: {missing_permissions}",
            tool_name=tool_name,
            error_code="PERMISSION_DENIED",
        )
        self.missing_permissions = missing_permissions


class ToolTenantMismatchError(ToolError):
    """Raised when tool arguments attempt to access or target a tenant other than the authoritative ToolContext."""

    def __init__(self, tool_name: str, attempted_tenant: str, authoritative_tenant: str):
        super().__init__(
            f"Tenant mismatch in tool '{tool_name}'. Attempted: '{attempted_tenant}', Authoritative: '{authoritative_tenant}'",
            tool_name=tool_name,
            error_code="TENANT_MISMATCH",
        )
        self.attempted_tenant = attempted_tenant
        self.authoritative_tenant = authoritative_tenant


class ToolParameterValidationError(ToolError):
    """Raised when tool arguments fail schema validation or exceed size limits."""

    def __init__(self, tool_name: str, details: str):
        super().__init__(
            f"Parameter validation failed for tool '{tool_name}': {details}",
            tool_name=tool_name,
            error_code="INVALID_PARAMETERS",
        )
        self.details = details


class ToolBudgetExceededError(ToolError):
    """Raised when tool call count or runtime budget is exhausted."""

    def __init__(self, message: str, budget_type: str, limit: float, current: float):
        super().__init__(message, error_code="TOOL_BUDGET_EXCEEDED")
        self.budget_type = budget_type
        self.limit = limit
        self.current = current


class ToolExecutionTimeoutError(ToolError):
    """Raised when a tool execution exceeds its timeout."""

    def __init__(self, tool_name: str, timeout_seconds: float):
        super().__init__(
            f"Tool '{tool_name}' timed out after {timeout_seconds}s",
            tool_name=tool_name,
            error_code="TOOL_TIMEOUT",
        )
        self.timeout_seconds = timeout_seconds


class ToolExecutionFailedError(ToolError):
    """Raised when a tool handler encounters an unhandled runtime exception."""

    def __init__(self, tool_name: str, original_error: str):
        super().__init__(
            f"Tool '{tool_name}' execution failed: {original_error}",
            tool_name=tool_name,
            error_code="EXECUTION_FAILED",
        )
        self.original_error = original_error
