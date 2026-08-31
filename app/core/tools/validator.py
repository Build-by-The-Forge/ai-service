import json
from typing import Any, Optional

from app.core.tools.contracts import (
    ToolDefinition,
    ToolExecutionPolicy,
    ToolInvocation,
)
from app.core.tools.errors import (
    ToolError,
    ToolParameterValidationError,
    ToolTenantMismatchError,
)


class ToolValidator:
    """
    Validates tool invocation arguments, tenant isolation boundaries, payload sizes, and schema requirements.
    """

    @staticmethod
    def validate_input(
        definition: ToolDefinition,
        invocation: ToolInvocation,
        policy: Optional[ToolExecutionPolicy] = None,
    ) -> None:
        # 1. Check context liveness & cancellation
        if invocation.context.is_cancelled():
            raise ToolError(
                f"Cannot execute tool '{definition.name}': execution context is cancelled",
                tool_name=definition.name,
                error_code="EXECUTION_CANCELLED",
            )
        if invocation.context.is_expired():
            raise ToolError(
                f"Cannot execute tool '{definition.name}': deadline expired",
                tool_name=definition.name,
                error_code="DEADLINE_EXPIRED",
            )

        # 2. Check input payload size limit
        if policy:
            try:
                payload_bytes = len(json.dumps(invocation.arguments).encode("utf-8"))
                if payload_bytes > policy.max_input_bytes:
                    raise ToolParameterValidationError(
                        definition.name,
                        f"Input payload size ({payload_bytes} bytes) exceeds limit ({policy.max_input_bytes} bytes)",
                    )
            except (TypeError, ValueError) as json_err:
                raise ToolParameterValidationError(
                    definition.name, f"Arguments could not be serialized to JSON: {str(json_err)}"
                )

        # 3. Tenant Boundary Assertion
        args = invocation.arguments
        for tenant_key in ("tenant_id", "school_id"):
            if tenant_key in args and args[tenant_key] is not None:
                attempted = str(args[tenant_key])
                authoritative = str(invocation.context.tenant_id)
                if attempted != authoritative:
                    raise ToolTenantMismatchError(
                        tool_name=definition.name,
                        attempted_tenant=attempted,
                        authoritative_tenant=authoritative,
                    )

        # 4. Schema validation (required properties)
        schema = definition.input_schema
        required_fields = schema.get("required", [])
        for field in required_fields:
            if field not in args:
                raise ToolParameterValidationError(
                    definition.name, f"Missing required property: '{field}'"
                )

    @staticmethod
    def validate_output(
        tool_name: str,
        data: Any,
        policy: Optional[ToolExecutionPolicy] = None,
    ) -> int:
        """
        Validates output payload size and returns byte count.
        """
        try:
            output_bytes = len(json.dumps(data).encode("utf-8"))
        except (TypeError, ValueError):
            output_bytes = len(str(data).encode("utf-8"))

        if policy and output_bytes > policy.max_output_bytes:
            raise ToolParameterValidationError(
                tool_name,
                f"Output payload size ({output_bytes} bytes) exceeds limit ({policy.max_output_bytes} bytes)",
            )

        return output_bytes
