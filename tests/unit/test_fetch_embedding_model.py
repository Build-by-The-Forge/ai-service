"""scripts/fetch_embedding_model.py: nothing is installed unless verified (P3-4)."""

import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "fetch_embedding_model", ROOT / "scripts" / "fetch_embedding_model.py"
)
fetcher = importlib.util.module_from_spec(_spec)
sys.modules["fetch_embedding_model"] = fetcher
_spec.loader.exec_module(fetcher)


def manifest(tmp_path, files):
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "schema": 1,
                "source": {"type": "huggingface", "repository": "org/model", "revision": "abc"},
                "files": [
                    {"name": n, "sha256": hashlib.sha256(b).hexdigest(), "size": len(b)}
                    for n, b in files.items()
                ],
            }
        )
    )
    return path


def test_the_real_manifest_pins_a_revision_and_verifiable_files():
    data = json.loads(fetcher.DEFAULT_MANIFEST.read_text(encoding="utf-8"))
    assert re.fullmatch(r"[0-9a-f]{40}", data["source"]["revision"])
    assert data["license"] == "Apache-2.0" and data["dimension"] == 384
    assert {f["name"] for f in data["files"]} == {"model.onnx", "tokenizer.json"}
    for f in data["files"]:
        assert re.fullmatch(r"[0-9a-f]{64}", f["sha256"]) and f["size"] > 0


def test_verified_cache_files_are_installed_without_a_download(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "model.onnx").write_bytes(b"weights")
    monkeypatch.setattr(fetcher, "download", lambda *a: pytest.fail("downloaded"))
    path = manifest(tmp_path, {"model.onnx": b"weights"})
    assert fetcher.fetch(path, tmp_path / "dest", cache, 1) == ["model.onnx"]
    assert (tmp_path / "dest" / "model.onnx").read_bytes() == b"weights"


def test_a_download_is_verified_and_a_mismatch_installs_nothing(tmp_path, monkeypatch):
    urls = []

    def fake_download(url, target, timeout):
        urls.append(url)
        target.write_bytes(b"tampered")

    monkeypatch.setattr(fetcher, "download", fake_download)
    path = manifest(tmp_path, {"model.onnx": b"weights"})
    with pytest.raises(fetcher.ModelFetchError, match="the manifest expects 7 bytes"):
        fetcher.fetch(path, tmp_path / "dest", None, 1)
    assert urls == ["https://huggingface.co/org/model/resolve/abc/model.onnx"]
    assert not (tmp_path / "dest" / "model.onnx").exists()


def test_a_corrupt_cache_file_is_ignored_and_fetched_again(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "model.onnx").write_bytes(b"corrupt")
    monkeypatch.setattr(fetcher, "download", lambda url, target, timeout: target.write_bytes(b"ok"))
    path = manifest(tmp_path, {"model.onnx": b"ok"})
    assert fetcher.fetch(path, tmp_path / "dest", cache, 1) == ["model.onnx"]


@pytest.mark.parametrize("name", ["../escape", ".hidden", "a/b"])
def test_unexpected_file_names_are_refused(tmp_path, name):
    with pytest.raises(fetcher.ModelFetchError, match="unexpected file name"):
        fetcher.fetch(manifest(tmp_path, {name: b"x"}), tmp_path / "dest", None, 1)


def test_main_reports_failure_with_a_non_zero_exit(tmp_path, monkeypatch, capsys):
    def boom(*a):
        raise OSError("network down")

    monkeypatch.setattr(fetcher, "download", boom)
    path = manifest(tmp_path, {"model.onnx": b"x"})
    assert fetcher.main(["--manifest", str(path), "--dest", str(tmp_path / "d")]) == 1
    assert "network down" in capsys.readouterr().err
