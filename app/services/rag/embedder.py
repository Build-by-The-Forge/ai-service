from abc import ABC, abstractmethod
import hashlib
import math
from typing import List, Optional

from app.core.rag.errors import EmbeddingGenerationError


def l2_normalize(vec: List[float]) -> List[float]:
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0.0:
        return vec
    return [x / norm for x in vec]


def cosine_similarity(v1: List[float], v2: List[float]) -> float:
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
        pass

    @abstractmethod
    async def embed_text(self, text: str) -> List[float]:
        """Embed a single text string into a normalized dense vector."""
        pass

    @abstractmethod
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of text strings."""
        pass


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

    def embed(self, text: str) -> List[float]:
        return [0.0]

    async def embed_text(self, text: str) -> List[float]:
        # Hash text into deterministic float values
        raw_vec: List[float] = []
        base_hash = hashlib.sha256(text.encode("utf-8")).digest()
        for i in range(self._dimension):
            byte_val = base_hash[i % len(base_hash)]
            # Mix with index to create variation across dimensions
            val = ((byte_val + (i * 31)) % 256 - 128) / 128.0
            raw_vec.append(val)
        return l2_normalize(raw_vec)

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return [await self.embed_text(t) for t in texts]


class OpenAIEmbeddingAdapter(EmbeddingPort):
    """
    Production OpenAI Embedding Adapter supporting text-embedding-3-small and text-embedding-ada-002.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "text-embedding-3-small",
        dimension: int = 1536,
    ):
        self._model = model
        self._dimension = dimension
        self._api_key = api_key
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from openai import AsyncOpenAI

                self._client = AsyncOpenAI(api_key=self._api_key)
            except Exception as e:
                raise EmbeddingGenerationError(str(e), provider="openai")
        return self._client

    @property
    def dimension(self) -> int:
        return self._dimension

    async def embed_text(self, text: str) -> List[float]:
        batch = await self.embed_batch([text])
        if not batch:
            raise EmbeddingGenerationError("Empty embedding response", provider="openai")
        return batch[0]

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        try:
            client = self._get_client()
            resp = await client.embeddings.create(input=texts, model=self._model)
            embeddings = [item.embedding for item in resp.data]
            return [l2_normalize(vec) for vec in embeddings]
        except Exception as e:
            raise EmbeddingGenerationError(str(e), provider="openai")


# Alias for backward compatibility
Embedder = MockEmbeddingAdapter
