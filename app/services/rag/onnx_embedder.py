"""Local text embeddings: all-MiniLM-L6-v2 on ONNX Runtime (P3-4, D3).

The same computation fastembed performs for this model (WordPiece
tokenization truncated at 256 tokens, the ONNX encoder, mean pooling over
the attention mask, L2 normalization), without fastembed's download and
runtime dependencies: only onnxruntime and tokenizers. No text leaves the
process.

The model loads lazily, once, on first use. Inference runs in a worker
thread so the event loop stays free, one batch at a time under a lock: the
session's threads already use the CPU quota, and concurrent runs would
oversubscribe it.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from app.core.rag.errors import EmbeddingGenerationError
from app.services.rag.embedder import EmbeddingPort

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
DIMENSION = 384
MAX_TOKENS = 256
PROVIDER = "onnx"


class OnnxMiniLMEmbedder(EmbeddingPort):
    def __init__(self, model_dir: str | Path, *, threads: int = 4, batch_size: int = 32) -> None:
        self._model_dir = Path(model_dir)
        self._threads = threads
        self._batch_size = batch_size
        self._lock = threading.Lock()
        self._session: Any = None
        self._tokenizer: Any = None

    @classmethod
    def from_settings(cls) -> OnnxMiniLMEmbedder:
        from app.core.config import settings

        if not settings.EMBEDDING_MODEL_DIR:
            raise EmbeddingGenerationError("EMBEDDING_MODEL_DIR is not set", provider=PROVIDER)
        return cls(
            settings.EMBEDDING_MODEL_DIR,
            threads=settings.EMBEDDING_THREADS,
            batch_size=settings.EMBEDDING_BATCH_SIZE,
        )

    @property
    def dimension(self) -> int:
        return DIMENSION

    @property
    def model_id(self) -> str:
        return MODEL_ID

    def _load(self) -> None:
        if self._session is not None:
            return
        model, vocab = self._model_dir / "model.onnx", self._model_dir / "tokenizer.json"
        if not (model.is_file() and vocab.is_file()):
            raise EmbeddingGenerationError(
                f"model files are missing from {self._model_dir}", provider=PROVIDER
            )
        import onnxruntime as ort
        from tokenizers import Tokenizer

        tokenizer = Tokenizer.from_file(str(vocab))
        tokenizer.enable_truncation(max_length=MAX_TOKENS)
        tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
        options = ort.SessionOptions()
        options.intra_op_num_threads = self._threads
        options.inter_op_num_threads = 1
        self._session = ort.InferenceSession(
            str(model), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self._tokenizer = tokenizer

    def _embed_sync(self, texts: Sequence[str]) -> np.ndarray:
        with self._lock:
            self._load()
            out = []
            for start in range(0, len(texts), self._batch_size):
                encoded = self._tokenizer.encode_batch(
                    list(texts[start : start + self._batch_size])
                )
                ids = np.array([e.ids for e in encoded], dtype=np.int64)
                mask = np.array([e.attention_mask for e in encoded], dtype=np.int64)
                (hidden,) = self._session.run(
                    None,
                    {
                        "input_ids": ids,
                        "attention_mask": mask,
                        "token_type_ids": np.zeros_like(ids),
                    },
                )
                weights = mask[..., None].astype(np.float32)
                pooled = (hidden * weights).sum(axis=1) / np.clip(weights.sum(axis=1), 1e-9, None)
                out.append(
                    pooled / np.clip(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-12, None)
                )
            return np.concatenate(out).astype(np.float32)

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            vectors = await asyncio.to_thread(self._embed_sync, texts)
        except EmbeddingGenerationError:
            raise
        except Exception as exc:  # onnxruntime and tokenizers raise their own types
            raise EmbeddingGenerationError(
                f"{type(exc).__name__}: {exc}", provider=PROVIDER
            ) from exc
        return vectors.tolist()

    async def embed_text(self, text: str) -> list[float]:
        return (await self.embed_batch([text]))[0]
