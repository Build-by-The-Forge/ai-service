"""Structured question generation (M5 design §3, ruling R2).

The model is asked for questions in the backend's question shape. Each item
is validated; anything malformed is discarded and counted, never repaired.
Retrieval context comes from the verified request tenant only.
"""

import logging
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from app.core.prompt_builder import PromptBuilder
from app.schemas.ai import GeneratedQuestion
from app.services.ai.llm import LLMProvider, StructuredOutputError
from app.services.rag.retrieval import RetrievalService

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GeneratedQuestions:
    questions: list[GeneratedQuestion]
    discarded: int
    # False when the tenant's material has nothing on the topic: no
    # questions are generated from the model's own knowledge.
    grounded: bool = True


class StructuredQuestionGenerator:
    def __init__(self, llm: LLMProvider, retriever: RetrievalService) -> None:
        self.llm = llm
        self.retriever = retriever
        self.prompts = PromptBuilder()

    async def generate(
        self, topic: str, *, count: int, difficulty: str, question_type: str
    ) -> GeneratedQuestions:
        evidence = await self.retriever.evidence(topic)
        if not evidence.passages:
            return GeneratedQuestions(questions=[], discarded=0, grounded=False)
        prompt = self.prompts.build(
            "structured_questions",
            topic=topic,
            count=count,
            difficulty=difficulty,
            question_type=question_type,
            context=evidence.prompt_block(),
        )
        try:
            answer = await self.llm.generate_json(
                prompt["system"] + "\n" + prompt["user"],
                kind="questions",
                count=count,
                difficulty=difficulty,
                question_type=question_type,
            )
        except StructuredOutputError:
            logger.warning("question generation: the model did not answer with JSON")
            return GeneratedQuestions(questions=[], discarded=count)

        raw = answer.get("questions")
        items: list[Any] = raw if isinstance(raw, list) else []
        valid: list[GeneratedQuestion] = []
        discarded = 0
        for item in items:
            try:
                question = GeneratedQuestion.model_validate(item)
            except ValidationError:
                discarded += 1
                continue
            # Only what was asked for: the requested type and difficulty.
            if question.question_type != question_type or question.difficulty != difficulty:
                discarded += 1
                continue
            valid.append(question)
        if len(valid) > count:
            discarded += len(valid) - count
            valid = valid[:count]
        if not isinstance(raw, list):
            discarded = count
        return GeneratedQuestions(questions=valid, discarded=discarded)
