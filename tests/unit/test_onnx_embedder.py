"""The local embedder (P3-4): pooling, batching and failures, without the model.

The real-model test runs where EMBEDDING_MODEL_DIR holds the verified files
(the image sets it); its exact agreement with fastembed was measured once on
200 evaluation chunks (max abs difference 1.2e-7).
"""

import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from app.core.rag.errors import EmbeddingGenerationError
from app.services.rag.onnx_embedder import DIMENSION, MODEL_ID, OnnxMiniLMEmbedder


class FakeTokenizer:
    """Token count = word count; padding to the longest text in the batch."""

    def encode_batch(self, texts):
        longest = max(len(t.split()) for t in texts)
        return [
            SimpleNamespace(
                ids=[1] * len(t.split()) + [0] * (longest - len(t.split())),
                attention_mask=[1] * len(t.split()) + [0] * (longest - len(t.split())),
            )
            for t in texts
        ]


class FakeSession:
    """Token j of every text has hidden state (j+1) on axis 0 and 1 on axis 1;
    padded positions carry a large value that pooling must ignore."""

    def __init__(self):
        self.batches = []

    def run(self, _, feeds):
        mask = feeds["attention_mask"]
        self.batches.append(mask.shape[0])
        hidden = np.zeros((*mask.shape, DIMENSION), dtype=np.float32)
        for j in range(mask.shape[1]):
            hidden[:, j, 0] = j + 1
            hidden[:, j, 1] = 1
        hidden[mask == 0] = 1000.0
        return [hidden]


def embedder(batch_size=32):
    e = OnnxMiniLMEmbedder("/nonexistent", threads=1, batch_size=batch_size)
    e._tokenizer, e._session = FakeTokenizer(), FakeSession()
    return e


async def test_mean_pools_real_tokens_only_and_normalizes():
    (vec,) = await embedder().embed_batch(["one two three"])
    # Mean of tokens 1..3 on axis 0 is 2, axis 1 is 1; padding is ignored.
    expected = np.zeros(DIMENSION)
    expected[0], expected[1] = 2, 1
    expected /= np.linalg.norm(expected)
    assert np.allclose(vec, expected, atol=1e-6)


async def test_padding_in_a_shared_batch_does_not_change_a_vector():
    alone = (await embedder().embed_batch(["a b"]))[0]
    padded = (await embedder().embed_batch(["a b", "a b c d e f"]))[0]
    assert np.allclose(alone, padded, atol=1e-6)


async def test_batches_are_bounded():
    e = embedder(batch_size=2)
    vectors = await e.embed_batch(["a", "b", "c", "d", "e"])
    assert len(vectors) == 5 and e._session.batches == [2, 2, 1]


async def test_empty_input_runs_nothing():
    e = embedder()
    assert await e.embed_batch([]) == [] and e._session.batches == []


async def test_missing_model_files_are_a_typed_error(tmp_path):
    with pytest.raises(EmbeddingGenerationError, match="model files are missing"):
        await OnnxMiniLMEmbedder(tmp_path).embed_text("x")


async def test_runtime_failures_are_a_typed_error():
    e = embedder()
    e._session = SimpleNamespace(run=lambda *_: (_ for _ in ()).throw(RuntimeError("boom")))
    with pytest.raises(EmbeddingGenerationError, match="RuntimeError: boom"):
        await e.embed_text("x")


def test_settings_without_a_model_directory_are_refused(monkeypatch):
    from app.core.config import settings

    monkeypatch.setattr(settings, "EMBEDDING_MODEL_DIR", None)
    with pytest.raises(EmbeddingGenerationError, match="EMBEDDING_MODEL_DIR is not set"):
        OnnxMiniLMEmbedder.from_settings()


MODEL_DIR = os.getenv("EMBEDDING_MODEL_DIR")


@pytest.mark.skipif(
    not (MODEL_DIR and (Path(MODEL_DIR) / "model.onnx").is_file()),
    reason="needs the verified model files (EMBEDDING_MODEL_DIR)",
)
async def test_real_model_ranks_the_related_text_first():
    e = OnnxMiniLMEmbedder(MODEL_DIR, threads=1)
    q, near, far = await e.embed_batch(
        [
            "Who first measured the charge of the electron?",
            "Millikan's oil drop experiment determined the elementary charge.",
            "The quotient rule differentiates a ratio of two functions.",
        ]
    )
    assert e.model_id == MODEL_ID and len(q) == DIMENSION
    assert np.dot(q, near) > np.dot(q, far) + 0.2
