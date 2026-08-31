from typing import Any

from app.core.rag.contracts import RetrievalFilter, RetrievalPolicy
from app.core.tools.contracts import ToolContext, ToolDefinition, ToolSideEffect
from app.services.rag.retrieval import get_retrieval_engine

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
            "course_code": {
                "type": "string",
                "description": "Optional course code filter (e.g. CSC201)",
            },
        },
        "required": ["query"],
    },
    output_schema={
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "tenant_id": {"type": "string"},
            "results_count": {"type": "integer"},
            "results": {"type": "array"},
        },
    },
    required_permissions=["content:read"],
    required_capabilities=["content_search"],
    tenant_scoped=True,
    side_effect=ToolSideEffect.READ_ONLY,
    timeout_seconds=15.0,
)


async def content_search_handler(context: ToolContext, arguments: dict[str, Any]) -> dict[str, Any]:
    """
    Authoritative handler for content.search tool.
    Executes live hybrid dense/sparse retrieval filtered strictly by the authoritative ToolContext.tenant_id.
    """
    query = str(arguments["query"])
    limit = int(arguments.get("limit", 3))
    course_code = arguments.get("course_code")

    engine = get_retrieval_engine()
    retrieval_filter = RetrievalFilter(
        tenant_id=context.tenant_id,
        course_code=str(course_code) if course_code else None,
    )
    retrieval_policy = RetrievalPolicy(top_k=max(1, min(limit, 10)))

    ctx = (
        await engine.retrieve(query=query, filter=retrieval_filter, policy=policy)
        if (policy := retrieval_policy)
        else None
    )

    if not ctx:
        return {
            "tenant_id": context.tenant_id,
            "query": query,
            "results_count": 0,
            "results": [],
        }

    results = []
    for rc in ctx.chunks:
        item: dict[str, Any] = {
            "chunk_id": rc.chunk.chunk_id,
            "document_title": rc.chunk.document_title,
            "course_code": rc.chunk.course_code,
            "topic": rc.chunk.topic,
            "snippet": rc.chunk.content,
            "relevance_score": round(rc.hybrid_score, 4),
            "rank": rc.rank,
        }
        if rc.chunk.provenance:
            item["provenance"] = rc.chunk.provenance.to_citation_dict()
        results.append(item)

    return {
        "tenant_id": context.tenant_id,
        "query": query,
        "results_count": len(results),
        "results": results,
    }
