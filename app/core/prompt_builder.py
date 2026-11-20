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

        if name == "tutor_turn":
            # M3-F: one conversation turn. The earlier questions only say
            # what the student means; they are never evidence. No memory.
            earlier = kwargs.get("earlier_questions") or []
            earlier_block = (
                "EARLIER QUESTIONS (context only, not evidence):\n"
                + "\n".join(f"- {q}" for q in earlier)
                + "\n\n"
                if earlier
                else ""
            )
            return {
                "system": """
You are an AI tutor for a university course, in a conversation with a student.

Answer ONLY from the numbered course passages provided. After each claim,
cite the passage it comes from as [n]. If the passages do not contain the
answer, say that the course materials do not cover it, and cite nothing.
Never add facts that are not in the passages. The student's earlier
questions only help you understand the current question; they are not a
source. Cite with [n] only: do not list sources, quote passage headers or
copy document identifiers into the answer.
""",
                "user": """
{earlier}COURSE PASSAGES:
{context}

QUESTION:
{question}
""".format(
                    earlier=earlier_block,
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
                # The course, when given, disambiguates the topic. Without one
                # the prompt is unchanged.
                "user": """
{course_line}TOPIC: {topic}
QUESTION TYPE: {question_type}
DIFFICULTY: {difficulty}
NUMBER OF QUESTIONS: {count}

CONTEXT:
{context}
""".format(
                    course_line=f"COURSE: {kwargs['course']}\n" if kwargs.get("course") else "",
                    topic=kwargs.get("topic", ""),
                    question_type=kwargs.get("question_type", "mcq"),
                    difficulty=kwargs.get("difficulty", "medium"),
                    count=kwargs.get("count", 5),
                    context=kwargs.get("context", ""),
                ),
            }

        return {"system": "", "user": kwargs.get("question", "")}
