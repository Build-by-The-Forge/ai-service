import uuid
from typing import Any

from app.core.tools.contracts import ToolContext, ToolDefinition, ToolSideEffect

DRAFT_QUESTION_DEFINITION = ToolDefinition(
    name="draft.question",
    version="1.0.0",
    description=(
        "Generate a structured multiple-choice or short-answer assessment item proposal. "
        "Produces an actionable draft proposal without mutating backend databases directly."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "topic": {
                "type": "string",
                "description": "Topic or curriculum concept to generate the question for",
            },
            "difficulty": {
                "type": "string",
                "enum": ["EASY", "MEDIUM", "HARD"],
                "description": "Target difficulty tier",
            },
            "question_type": {
                "type": "string",
                "enum": ["MULTIPLE_CHOICE", "SHORT_ANSWER"],
                "default": "MULTIPLE_CHOICE",
            },
        },
        "required": ["topic", "difficulty"],
    },
    output_schema={
        "type": "object",
        "properties": {
            "proposal_id": {"type": "string"},
            "topic": {"type": "string"},
            "difficulty": {"type": "string"},
            "question_stem": {"type": "string"},
            "options": {"type": "array"},
            "correct_answer": {"type": "string"},
            "explanation": {"type": "string"},
        },
    },
    required_permissions=["assessment:generate"],
    required_capabilities=["cbt_generation"],
    tenant_scoped=True,
    side_effect=ToolSideEffect.PROPOSAL,
    timeout_seconds=20.0,
)


async def draft_question_handler(context: ToolContext, arguments: dict[str, Any]) -> dict[str, Any]:
    topic = arguments["topic"]
    difficulty = arguments["difficulty"]
    q_type = arguments.get("question_type", "MULTIPLE_CHOICE")

    proposal_id = f"prop_{uuid.uuid4().hex[:12]}"

    return {
        "status": "PROPOSAL_CREATED",
        "proposal_id": proposal_id,
        "tenant_id": context.tenant_id,
        "topic": topic,
        "difficulty": difficulty,
        "question_type": q_type,
        "question_stem": f"Which of the following best describes {topic}?",
        "options": [
            {"label": "A", "text": f"Primary principle of {topic}"},
            {"label": "B", "text": "Incorrect distractor option 1"},
            {"label": "C", "text": "Incorrect distractor option 2"},
            {"label": "D", "text": "Incorrect distractor option 3"},
        ],
        "correct_answer": "A",
        "explanation": f"Option A accurately characterizes the core definition of {topic}.",
        "disclaimer": "AI Proposal: Requires human review and Backend commit before student assignment.",
    }
