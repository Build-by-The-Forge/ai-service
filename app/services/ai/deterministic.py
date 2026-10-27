import hashlib
import re
from typing import Any

from app.services.ai.llm import LLMProvider

_PASSAGE_ONE = re.compile(r"\n\[1\] \([^)]*\) (.*?)(?:\n\n\[2\] |\Z)", re.S)
_WORD = re.compile(r"[^\W\d_]{4,}")


def _first_passage_words(prompt: str) -> list[str]:
    """Words (no numbers) of the first numbered course passage in a prompt."""
    match = _PASSAGE_ONE.search(prompt)
    return _WORD.findall(match.group(1)) if match else []


class DeterministicLLM(LLMProvider):
    """Offline stand-in for a model (MODEL_PROVIDER=deterministic).

    No network and no model: the answer names a digest and the size of the
    prompt, so a test can see that a request reached the model layer and with
    which prompt, without any real tutoring. Development and testing only.
    """

    async def generate(self, prompt: str) -> str:
        digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]
        answer = f"[deterministic model] prompt sha256:{digest} ({len(prompt)} chars)"
        # Behaves as a compliant model under the grounding rules: it cites
        # the first numbered course passage when it was given passages.
        if "\n[1] (" in prompt:
            answer += " [1]"
        return answer

    async def generate_json(self, prompt: str, *, kind: str, **hints: Any) -> dict[str, Any]:
        """Well-formed structured answers for the kinds AI-service asks for,
        labelled as deterministic so they are never mistaken for content."""
        digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:8]
        if kind == "questions":
            count = int(hints.get("count", 1))
            true_false = hints.get("question_type") == "true_false"
            # As a compliant model would: questions in the words of passage
            # [1], citing it, so they pass the support checks.
            words = _first_passage_words(prompt)
            questions = []
            for i in range(1, count + 1):
                stem = " ".join(words[(i - 1) % max(1, len(words)) :][:8]) or f"Question {i}"
                answer = " ".join(words[8:12]) or "Option A"
                if true_false:
                    options = [
                        {"text": "True", "is_correct": i % 2 == 1},
                        {"text": "False", "is_correct": i % 2 == 0},
                    ]
                else:
                    options = [{"text": answer, "is_correct": True}] + [
                        {"text": f"Option {c}", "is_correct": False} for c in "BCD"
                    ]
                questions.append(
                    {
                        "question_type": "true_false" if true_false else "mcq",
                        "prompt": f"[deterministic question] {stem}",
                        "options": options,
                        "explanation": f"[deterministic {digest}] Explanation {i}",
                        "difficulty": hints.get("difficulty", "medium"),
                        "sources": [1] if words else [],
                    }
                )
            return {"questions": questions}
        return {"answer": await self.generate(prompt)}
