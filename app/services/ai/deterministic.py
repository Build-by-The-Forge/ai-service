import hashlib

from app.services.ai.llm import LLMProvider


class DeterministicLLM(LLMProvider):
    """Offline stand-in for a model (MODEL_PROVIDER=deterministic).

    No network and no model: the answer names a digest and the size of the
    prompt, so a test can see that a request reached the model layer and with
    which prompt, without any real tutoring. Development and testing only.
    """

    async def generate(self, prompt: str) -> str:
        digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]
        return f"[deterministic model] prompt sha256:{digest} ({len(prompt)} chars)"
