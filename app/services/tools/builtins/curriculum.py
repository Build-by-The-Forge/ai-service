from typing import Any, Dict

from app.core.tools.contracts import ToolContext, ToolDefinition, ToolSideEffect

CURRICULUM_LOOKUP_DEFINITION = ToolDefinition(
    name="curriculum.lookup",
    version="1.0.0",
    description="Look up academic curriculum courses, unit structures, and topic outlines for the school tenant.",
    input_schema={
        "type": "object",
        "properties": {
            "course_code": {
                "type": "string",
                "description": "Standard course code (e.g. 'CSC201', 'PHY101')",
            },
            "level": {
                "type": "integer",
                "description": "Academic level (e.g. 100, 200, 300)",
            },
        },
        "required": ["course_code"],
    },
    output_schema={
        "type": "object",
        "properties": {
            "course_code": {"type": "string"},
            "title": {"type": "string"},
            "units": {"type": "array"},
        },
    },
    required_permissions=["curriculum:read"],
    required_capabilities=["curriculum_intelligence"],
    tenant_scoped=True,
    side_effect=ToolSideEffect.READ_ONLY,
    timeout_seconds=10.0,
)


async def curriculum_lookup_handler(
    context: ToolContext, arguments: Dict[str, Any]
) -> Dict[str, Any]:
    course_code = arguments["course_code"].upper()
    return {
        "tenant_id": context.tenant_id,
        "course_code": course_code,
        "title": f"Introductory {course_code}",
        "units": [
            {
                "unit_id": f"{course_code}_U1",
                "title": "Fundamental Concepts",
                "topics": ["Introduction", "Core Principles"],
            },
            {
                "unit_id": f"{course_code}_U2",
                "title": "Advanced Applications",
                "topics": ["Applied Theory", "Case Studies"],
            },
        ],
    }
