"""benchmarks/offline/fetch_model.py and models.json: pinned, verified, never partial."""

import hashlib
import importlib.util
import io
import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "benchmarks" / "offline"
_spec = importlib.util.spec_from_file_location("offline_fetch", BENCH / "fetch_model.py")
fetcher = importlib.util.module_from_spec(_spec)
sys.modules["offline_fetch"] = fetcher
_spec.loader.exec_module(fetcher)

REGISTRY = json.loads((BENCH / "models.json").read_text(encoding="utf-8"))
PAYLOAD = b"GGUF" + bytes(range(256)) * 4096


def model(payload=PAYLOAD):
    return {
        "repo": "org/model-GGUF",
        "revision": "a" * 40,
        "file": "m.gguf",
        "size": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


class Response(io.BytesIO):
    """urlopen() stand-in; can fail after the first block to simulate a dropped link."""

    def __init__(self, data, fail_after_first=False):
        super().__init__(data)
        self.fail_after_first, self.reads = fail_after_first, 0

    def read(self, size=-1):
        self.reads += 1
        if self.fail_after_first and self.reads > 1:
            raise ConnectionResetError("link dropped")
        return super().read(size)


def opener(data, **kwargs):
    seen = []

    def open_(url, timeout):
        seen.append(url)
        return Response(data, **kwargs)

    open_.seen = seen
    return open_


def test_registry_entries_are_pinned_and_complete():
    ids = [m["id"] for m in REGISTRY["models"]]
    assert len(ids) == len(set(ids))
    for m in REGISTRY["models"]:
        assert re.fullmatch(r"[0-9a-f]{40}", m["revision"]), m["id"]
        assert re.fullmatch(r"[0-9a-f]{64}", m["sha256"]), m["id"]
        assert m["size"] > 100_000_000, m["id"]
        assert m["card"].startswith("https://huggingface.co/"), m["id"]
        assert m["licence"] and m["runtime"] in ("llama.cpp", "litert-lm"), m["id"]


def test_download_is_verified_and_uses_the_pinned_revision(tmp_path):
    open_ = opener(PAYLOAD)
    assert fetcher.fetch(model(), tmp_path, open_) == 0
    assert (tmp_path / "m.gguf").read_bytes() == PAYLOAD
    assert open_.seen == [f"https://huggingface.co/org/model-GGUF/resolve/{'a' * 40}/m.gguf"]
    # Second call is a no-op: already present and verified.
    assert fetcher.fetch(model(), tmp_path, opener(b"")) == 0


def test_corrupted_download_is_deleted(tmp_path):
    corrupted = bytearray(PAYLOAD)
    corrupted[100] ^= 0xFF
    assert fetcher.fetch(model(), tmp_path, opener(bytes(corrupted))) == 1
    assert list(tmp_path.iterdir()) == []


def test_truncated_download_is_deleted(tmp_path):
    assert fetcher.fetch(model(), tmp_path, opener(PAYLOAD[:-10])) == 1
    assert list(tmp_path.iterdir()) == []


def test_interrupted_download_leaves_nothing_behind(tmp_path):
    with pytest.raises(ConnectionResetError):
        fetcher.fetch(model(), tmp_path, opener(PAYLOAD, fail_after_first=True))
    assert list(tmp_path.iterdir()) == []


def test_a_stale_wrong_file_is_replaced(tmp_path):
    (tmp_path / "m.gguf").write_bytes(b"old")
    assert fetcher.fetch(model(), tmp_path, opener(PAYLOAD)) == 0
    assert (tmp_path / "m.gguf").read_bytes() == PAYLOAD
