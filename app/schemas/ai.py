"""Request bodies of the signed AI routes under /v1/ai (design B.4, B.5).

None of them has a tenant, user, membership or role field: identity reaches
AI-service only in the signed context. Unknown fields are rejected (422), so a
body that tries to name a tenant or user fails instead of being ignored (C29).
"""

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.tree import TreeNode

MAX_MESSAGE_CHARS = 4000
MAX_QUESTION_COUNT = 50
MAX_TITLE_CHARS = 300
MAX_TREE_TOPICS = 200


class SignedAIRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ChatRequest(SignedAIRequest):
    message: str = Field(min_length=1, max_length=MAX_MESSAGE_CHARS)


class ExplainRequest(SignedAIRequest):
    subject: str = Field(min_length=1, max_length=MAX_MESSAGE_CHARS)


class GenerateQuestionsRequest(SignedAIRequest):
    topic: str = Field(min_length=1, max_length=MAX_MESSAGE_CHARS)
    count: int = Field(default=5, ge=1, le=MAX_QUESTION_COUNT)


class RetrieveRequest(SignedAIRequest):
    query: str = Field(min_length=1, max_length=MAX_MESSAGE_CHARS)


class TreeGenerateRequest(SignedAIRequest):
    course_title: str = Field(min_length=1, max_length=MAX_TITLE_CHARS)
    topics: list[str] = Field(max_length=MAX_TREE_TOPICS)


class ReadStartRequest(SignedAIRequest):
    root: TreeNode
