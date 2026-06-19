from typing import Dict, Any


class PromptBuilder:

    def build(self, name: str, **kwargs: Any) -> Dict[str, str]:
        if name == "chat_tutor":
            return {
                "system": """
You are a CBT AI tutor.

You MUST personalize responses based on student memory.

Rules:
- Use context + memory
- Adapt difficulty to mastery level
- Reinforce weak topics
""",
                "user": """
STUDENT MEMORY:
{memory}

CONTEXT:
{context}

QUESTION:
{question}
""".format(
                    memory=kwargs.get("memory", ""),
                    context=kwargs.get("context", ""),
                    question=kwargs.get("question", ""),
                ),
            }

        if name == "question_generator":
            return {
                "system": """
You are an expert CBT exam question generator.

You MUST base questions strictly on the provided context.

Rules:
- Do not hallucinate facts outside context
- Generate realistic CBT-style questions
- Include 4 options (A–D)
- Include correct answers
- Include difficulty level
- Ensure academic correctness
""",
                "user": """
COURSE: {topic}
DIFFICULTY: {difficulty}
NUMBER OF QUESTIONS: {count}

CONTEXT:
{context}

Generate structured CBT questions.
""".format(
                    topic=kwargs.get("topic", ""),
                    difficulty=kwargs.get("difficulty", "medium"),
                    count=kwargs.get("count", 5),
                    context=kwargs.get("context", ""),
                ),
            }

        return {"system": "", "user": kwargs.get("question", "")}
