"""Request bodies of the signed AI routes under /v1/ai (design B.4, B.5).

None of them has a tenant, user, membership or role field: identity reaches
AI-service only in the signed context. Unknown fields are rejected (422), so a
body that tries to name a tenant or user fails instead of being ignored (C29).
"""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.schemas.tree import TreeNode

MAX_MESSAGE_CHARS = 4000
MAX_QUESTION_COUNT = 50
MAX_TITLE_CHARS = 300
MAX_TREE_TOPICS = 200
MAX_MEMORY_TOPIC_CHARS = 255
MAX_MEMORY_QUERY_TOPICS = 50
MAX_MEMORY_OBSERVATIONS = 200


class SignedAIRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ChatRequest(SignedAIRequest):
    message: str = Field(min_length=1, max_length=MAX_MESSAGE_CHARS)


class ExplainRequest(SignedAIRequest):
    subject: str = Field(min_length=1, max_length=MAX_MESSAGE_CHARS)


Difficulty = Literal["easy", "medium", "hard"]
QuestionType = Literal["mcq", "true_false"]


class GenerateQuestionsRequest(SignedAIRequest):
    topic: str = Field(min_length=1, max_length=MAX_MESSAGE_CHARS)
    count: int = Field(default=5, ge=1, le=MAX_QUESTION_COUNT)
    difficulty: Difficulty = "medium"
    question_type: QuestionType = "mcq"


class RetrieveRequest(SignedAIRequest):
    query: str = Field(min_length=1, max_length=MAX_MESSAGE_CHARS)


class TreeGenerateRequest(SignedAIRequest):
    course_title: str = Field(min_length=1, max_length=MAX_TITLE_CHARS)
    topics: list[str] = Field(max_length=MAX_TREE_TOPICS)


class ReadStartRequest(SignedAIRequest):
    root: TreeNode


MemoryTopic = Annotated[str, Field(min_length=1, max_length=MAX_MEMORY_TOPIC_CHARS)]


class MemoryQueryRequest(SignedAIRequest):
    topics: list[MemoryTopic] | None = Field(default=None, max_length=MAX_MEMORY_QUERY_TOPICS)


class MemoryObservation(SignedAIRequest):
    topic: MemoryTopic
    correct: bool


class MemoryUpsertRequest(SignedAIRequest):
    observations: list[MemoryObservation] = Field(min_length=1, max_length=MAX_MEMORY_OBSERVATIONS)


# --- Responses (M5): the `data` of each route, so OpenAPI is the contract --------------


class ContextItem(BaseModel):
    text: str


class ChatData(BaseModel):
    answer: str
    context_used: list[ContextItem]
    memory_used: str


class ExplainData(BaseModel):
    explanation: str


class GeneratedOption(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1, max_length=1000)
    is_correct: bool


class GeneratedQuestion(BaseModel):
    """One generated question in the backend's question shape (M5 design §3).
    Anything the model returns that breaks these rules is discarded."""

    model_config = ConfigDict(extra="forbid")

    question_type: QuestionType
    prompt: str = Field(min_length=1, max_length=MAX_MESSAGE_CHARS)
    options: list[GeneratedOption]
    explanation: str | None = Field(default=None, max_length=MAX_MESSAGE_CHARS)
    difficulty: Difficulty

    @model_validator(mode="after")
    def _options_fit_the_type(self) -> "GeneratedQuestion":
        if sum(option.is_correct for option in self.options) != 1:
            raise ValueError("exactly one option must be correct")
        if self.question_type == "true_false":
            if sorted(option.text for option in self.options) != ["False", "True"]:
                raise ValueError("true_false options must be exactly True and False")
        elif not 2 <= len(self.options) <= 6:
            raise ValueError("mcq needs 2 to 6 options")
        if len({option.text.strip().lower() for option in self.options}) != len(self.options):
            raise ValueError("options must be distinct")
        return self


class GeneratedQuestionsData(BaseModel):
    questions: list[GeneratedQuestion]
    discarded: int


class _Envelope(BaseModel):
    success: bool = True
    message: str = ""


class ChatResponse(_Envelope):
    data: ChatData


class ExplainResponse(_Envelope):
    data: ExplainData


class GeneratedQuestionsResponse(_Envelope):
    data: GeneratedQuestionsData


class MemoryItem(BaseModel):
    topic: str
    mastery_score: float
    attempts: int


class MemoryData(BaseModel):
    items: list[MemoryItem]


class MemoryResponse(_Envelope):
    data: MemoryData
