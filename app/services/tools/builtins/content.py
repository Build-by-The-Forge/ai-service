from typing import Any, Dict

from app.core.tools.contracts import ToolContext, ToolDefinition, ToolSideEffect

CONTENT_SEARCH_DEFINITION = ToolDefinition(
    name="content.search",
    version="1.0.0",
    description="Search verified course materials, uploaded textbook packs, and lecture notes for the school tenant.",
    input_schema={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Natural language search query across indexed materials",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of search results to return",
                "default": 3,
            },
        },
        "required": ["query"],
    },
    output_schema={
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "results": {"type": "array"},
        },
    },
    required_permissions=["content:read"],
    required_capabilities=["content_search"],
    tenant_scoped=True,
    side_effect=ToolSideEffect.READ_ONLY,
    timeout_seconds=15.0,
)


async def content_search_handler(context: ToolContext, arguments: Dict[str, Any]) -> Dict[str, Any]:
    query = arguments["query"]
    limit = int(arguments.get("limit", 3))

    return {
        "tenant_id": context.tenant_id,
        "query": query,
        "results_count": min(limit, 2),
        "results": [
            {
                "chunk_id": "chunk_001",
                "document_title": "Lecture Notes - Chapter 1",
                "snippet": f"Content matching query '{query}' from Chapter 1.",
                "relevance_score": 0.92,
            },
            {
                "chunk_id": "chunk_002",
                "document_title": "Course Syllabus",
                "snippet": f"Additional context regarding '{query}'.",
                "relevance_score": 0.85,
            },
        ][:limit],
    }
