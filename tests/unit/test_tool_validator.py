import pytest

from app.core.tools.contracts import (
    ToolContext,
    ToolDefinition,
    ToolExecutionPolicy,
    ToolInvocation,
)
from app.core.tools.errors import (
    ToolParameterValidationError,
    ToolTenantMismatchError,
)
from app.core.tools.validator import ToolValidator


@pytest.fixture
def sample_definition():
    return ToolDefinition(
        name="test.lookup",
        version="1.0.0",
        description="Test lookup",
        input_schema={
            "type": "object",
            "properties": {
                "course_id": {"type": "string"},
                "tenant_id": {"type": "string"},
            },
            "required": ["course_id"],
        },
    )


@pytest.fixture
def sample_context():
    return ToolContext(
        tenant_id="school_authoritative_1",
        execution_id="exec_1",
        task_id="task_1",
        correlation_id="corr_1",
    )


def test_validator_detects_missing_required_properties(sample_definition, sample_context):
    invocation = ToolInvocation(
        tool_name="test.lookup",
        arguments={},  # missing required 'course_id'
        context=sample_context,
    )

    with pytest.raises(ToolParameterValidationError) as exc_info:
        ToolValidator.validate_input(sample_definition, invocation)
    assert "Missing required property: 'course_id'" in str(exc_info.value)


def test_validator_rejects_tenant_mismatch(sample_definition, sample_context):
    invocation = ToolInvocation(
        tool_name="test.lookup",
        arguments={"course_id": "CSC101", "tenant_id": "school_spoofed_99"},
        context=sample_context,  # Authoritative is school_authoritative_1
    )

    with pytest.raises(ToolTenantMismatchError) as exc_info:
        ToolValidator.validate_input(sample_definition, invocation)
    assert exc_info.value.attempted_tenant == "school_spoofed_99"
    assert exc_info.value.authoritative_tenant == "school_authoritative_1"


def test_validator_rejects_oversized_input_payload(sample_definition, sample_context):
    large_payload = {"course_id": "A" * 1000}
    invocation = ToolInvocation(
        tool_name="test.lookup",
        arguments=large_payload,
        context=sample_context,
    )
    policy = ToolExecutionPolicy(max_input_bytes=100)

    with pytest.raises(ToolParameterValidationError) as exc_info:
        ToolValidator.validate_input(sample_definition, invocation, policy)
    assert "Input payload size" in str(exc_info.value)


def test_validator_rejects_oversized_output_payload():
    large_output = {"data": "X" * 500}
    policy = ToolExecutionPolicy(max_output_bytes=100)

    with pytest.raises(ToolParameterValidationError) as exc_info:
        ToolValidator.validate_output("test.lookup", large_output, policy)
    assert "Output payload size" in str(exc_info.value)
