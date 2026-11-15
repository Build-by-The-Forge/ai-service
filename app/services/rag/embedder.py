import hashlib
import math
from abc import ABC, abstractmethod


def l2_normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0.0:
        return vec
    return [x / norm for x in vec]


def cosine_similarity(v1: list[float], v2: list[float]) -> float:
    if len(v1) != len(v2) or not v1:
        return 0.0
    dot = sum(a * b for a, b in zip(v1, v2))
    norm_a = math.sqrt(sum(a * a for a in v1))
    norm_b = math.sqrt(sum(b * b for b in v2))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return max(-1.0, min(1.0, dot / (norm_a * norm_b)))


class EmbeddingPort(ABC):
    """
    Abstract Port for vector embedding generation.
    """

    @property
    @abstractmethod
    def dimension(self) -> int:
        """The dimensionality of the output vector (e.g. 1536)."""

    @abstractmethod
    async def embed_text(self, text: str) -> list[float]:
        """Embed a single text string into a normalized dense vector."""

    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of text strings."""


class MockEmbeddingAdapter(EmbeddingPort):
    """
    Deterministic, zero-network mock embedding adapter for fast testing and development.
    Generates reproducible, normalized vectors using SHA256 hashing.
    """

    def __init__(self, dimension: int = 1536):
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, text: str) -> list[float]:
        return [0.0]

    async def embed_text(self, text: str) -> list[float]:
        # Hash text into deterministic float values
        raw_vec: list[float] = []
        base_hash = hashlib.sha256(text.encode("utf-8")).digest()
        for i in range(self._dimension):
            byte_val = base_hash[i % len(base_hash)]
            # Mix with index to create variation across dimensions
            val = ((byte_val + (i * 31)) % 256 - 128) / 128.0
            raw_vec.append(val)
        return l2_normalize(raw_vec)

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [await self.embed_text(t) for t in texts]


# Alias for backward compatibility
Embedder = MockEmbeddingAdapter
