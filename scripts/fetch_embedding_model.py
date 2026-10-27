"""Fetch, verify and install the embedding model listed in models/embedding/manifest.json.

    python scripts/fetch_embedding_model.py --dest /opt/build-ai/models/embedding \
        [--cache-dir models/embedding/files]

Every file is checked against the manifest's size and SHA-256 before it is
installed. A mismatch or a truncated download stops the run with a non-zero
exit; nothing ever falls back to an unverified file. Files come from the
pinned revision of the manifest's Hugging Face repository.

--cache-dir is read first: verified files found there are used without a
download (a read-only bind mount in the image build is fine).

Standard library only, so it runs in a bare build stage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
import urllib.request
from pathlib import Path

DEFAULT_MANIFEST = Path(__file__).resolve().parents[1] / "models" / "embedding" / "manifest.json"
CHUNK = 1024 * 1024


class ModelFetchError(Exception):
    """The model could not be obtained and verified."""


def digest(path: Path) -> tuple[int, str]:
    sha = hashlib.sha256()
    size = 0
    with path.open("rb") as fh:
        while block := fh.read(CHUNK):
            sha.update(block)
            size += len(block)
    return size, sha.hexdigest()


def verified(path: Path, entry: dict) -> bool:
    return path.is_file() and digest(path) == (int(entry["size"]), entry["sha256"].lower())


def download(url: str, target: Path, timeout: float) -> None:
    with (
        urllib.request.urlopen(url, timeout=timeout) as response,  # nosec B310 - manifest https URL
        target.open("wb") as fh,
    ):
        shutil.copyfileobj(response, fh, CHUNK)


def fetch(manifest_path: Path, dest: Path, cache_dir: Path | None, timeout: float) -> list[str]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != 1 or manifest["source"]["type"] != "huggingface":
        raise ModelFetchError(f"unsupported manifest in {manifest_path}")
    repo, revision = manifest["source"]["repository"], manifest["source"]["revision"]
    dest.mkdir(parents=True, exist_ok=True)
    installed = []
    with tempfile.TemporaryDirectory() as tmp:
        for entry in manifest["files"]:
            name = entry["name"]
            if "/" in name or name.startswith("."):
                raise ModelFetchError(f"unexpected file name in the manifest: {name!r}")
            cached = cache_dir / name if cache_dir else None
            if cached is not None and verified(cached, entry):
                source = cached
            else:
                source = Path(tmp) / name
                download(
                    f"https://huggingface.co/{repo}/resolve/{revision}/{name}", source, timeout
                )
                if not verified(source, entry):
                    size, sha = digest(source)
                    raise ModelFetchError(
                        f"{name}: got {size} bytes, sha256 {sha}; the manifest expects "
                        f"{entry['size']} bytes, sha256 {entry['sha256']}"
                    )
            shutil.copyfile(source, dest / name)
            installed.append(name)
    return installed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--dest", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args(argv)
    try:
        names = fetch(args.manifest, args.dest, args.cache_dir, args.timeout)
    except (ModelFetchError, OSError) as exc:
        print(f"embedding model fetch failed: {exc}", file=sys.stderr)
        return 1
    print(f"installed and verified: {', '.join(names)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
