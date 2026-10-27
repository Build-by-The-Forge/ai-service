from typing import Any


class PromptBuilder:
    def build(self, name: str, **kwargs: Any) -> dict[str, str]:
        if name == "chat_tutor":
            return {
                "system": """
You are a CBT AI tutor for a university course.

Answer ONLY from the numbered course passages provided. After each claim,
cite the passage it comes from as [n]. If the passages do not contain the
answer, say that the course materials do not cover it, and cite nothing.
Never add facts that are not in the passages.

Personalise the explanation with the student memory:
- Adapt difficulty to mastery level
- Reinforce weak topics
""",
                "user": """
STUDENT MEMORY:
{memory}

COURSE PASSAGES:
{context}

QUESTION:
{question}
""".format(
                    memory=kwargs.get("memory", ""),
                    context=kwargs.get("context", ""),
                    question=kwargs.get("question", ""),
                ),
            }

        if name == "explain":
            return {
                "system": """
You are an expert tutoring assistant for a university course.

Explain the subject in clear academic terms, using ONLY the numbered course
passages provided, and cite the passage behind each point as [n]. If the
passages do not cover the subject, say so and cite nothing. Never add facts
that are not in the passages.
""",
                "user": """
COURSE PASSAGES:
{context}

SUBJECT:
{subject}
""".format(
                    context=kwargs.get("context", ""),
                    subject=kwargs.get("subject", ""),
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

        if name == "structured_questions":
            return {
                "system": """
You write assessment questions for a university course.

Base every question strictly on the numbered course passages provided.
Answer with ONE JSON object: {"questions": [ ... ]}. Each question is:
{"question_type": "<type>", "prompt": "...",
 "options": [{"text": "...", "is_correct": true|false}, ...],
 "explanation": "why the correct option is correct",
 "difficulty": "<difficulty>",
 "sources": [<numbers of the passages the question and its answer come from>]}
Rules: exactly one option has is_correct true. For "mcq" give 4 options.
For "true_false" give exactly two options, "True" and "False".
Every question must be answerable from the passages it cites, and must not
introduce facts, names or numbers that are not in them.
""",
                "user": """
TOPIC: {topic}
QUESTION TYPE: {question_type}
DIFFICULTY: {difficulty}
NUMBER OF QUESTIONS: {count}

CONTEXT:
{context}
""".format(
                    topic=kwargs.get("topic", ""),
                    question_type=kwargs.get("question_type", "mcq"),
                    difficulty=kwargs.get("difficulty", "medium"),
                    count=kwargs.get("count", 5),
                    context=kwargs.get("context", ""),
                ),
            }

        return {"system": "", "user": kwargs.get("question", "")}
