"""Download one registered model file at its pinned revision and verify size and sha256.

    python benchmarks/offline/fetch_model.py qwen3.5-0.8b-q4km --dest /models

A file that fails verification is deleted. Weights never belong in the repository.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path
from typing import Any, Callable

REGISTRY = Path(__file__).resolve().parent / "models.json"


def entry(model_id: str) -> dict[str, Any]:
    models = json.loads(REGISTRY.read_text(encoding="utf-8"))["models"]
    for model in models:
        if model["id"] == model_id:
            return model
    raise SystemExit(f"unknown model id {model_id}; see {REGISTRY.name}")


def verify(path: Path, size: int, sha256: str) -> bool:
    if not path.exists() or path.stat().st_size != size:
        return False
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest() == sha256


def fetch(
    model: dict[str, Any], dest: Path, opener: Callable[..., Any] = urllib.request.urlopen
) -> int:
    """Download one registry entry into dest. Returns a process exit code."""
    target = dest / str(model["file"])
    size, sha = int(model["size"]), str(model["sha256"])
    if verify(target, size, sha):
        print(f"{target} already present and verified")
        return 0
    url = f"https://huggingface.co/{model['repo']}/resolve/{model['revision']}/{model['file']}"
    dest.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    print(f"downloading {url} ({size / 1e6:.0f} MB)", flush=True)
    try:
        with opener(url, timeout=60) as response, partial.open("wb") as out:
            while block := response.read(1 << 20):
                out.write(block)
    except BaseException:
        # An interrupted download never leaves a file that looks complete.
        partial.unlink(missing_ok=True)
        raise
    partial.replace(target)
    if not verify(target, size, sha):
        target.unlink()
        print("verification failed; file deleted", file=sys.stderr)
        return 1
    print(f"{target} verified (sha256 {sha[:12]}...)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_id")
    parser.add_argument("--dest", type=Path, required=True)
    args = parser.parse_args()
    return fetch(entry(args.model_id), args.dest)


if __name__ == "__main__":
    raise SystemExit(main())
