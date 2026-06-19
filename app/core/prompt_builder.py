from typing import Dict


class PromptBuilder:

    def build(self, name: str, context: str, question: str, memory: str) -> Dict[str, str]:
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
""".format(memory=memory, context=context, question=question),
            }

        return {"system": "", "user": question}
