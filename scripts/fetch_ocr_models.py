"""Fetch, verify and install the PaddleOCR models listed in models/ocr/manifest.json.

    python scripts/fetch_ocr_models.py --dest /opt/build-ai/models/paddleocr \
        [--archive-dir models/ocr/archives] [--source mirror|upstream]

Every archive is checked against the manifest's size and SHA-256 before
anything is extracted. A mismatch, a truncated download, a corrupt cached
archive or an unexpected archive member stops the run with a non-zero exit;
nothing ever falls back to an unverified file.

Sources:
  mirror (default)  the project-controlled GitHub Release named in the manifest.
                    A private repository needs a token in GITHUB_TOKEN (pass it
                    as a BuildKit secret; it is never written to an image layer).
  upstream          the original PaddleOCR URLs, only when asked for explicitly.

--archive-dir is a cache: verified archives found there are used without a
download, and downloads are kept there after verification.

Standard library only, so it runs in a bare build stage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tarfile
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

MODEL_FILES = frozenset({"inference.pdmodel", "inference.pdiparams", "inference.pdiparams.info"})
DEFAULT_MANIFEST = Path(__file__).resolve().parents[1] / "models" / "ocr" / "manifest.json"
CHUNK = 1024 * 1024


class ModelFetchError(Exception):
    """A model could not be obtained and verified."""


@dataclass(frozen=True)
class ModelEntry:
    role: str
    name: str
    archive: str
    sha256: str
    size: int
    upstream_url: str


def load_manifest(path: Path) -> tuple[dict, list[ModelEntry]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema") != 1:
        raise ModelFetchError(f"unsupported manifest schema in {path}")
    models = [
        ModelEntry(
            role=m["role"],
            name=m["name"],
            archive=m["archive"],
            sha256=m["sha256"].lower(),
            size=int(m["size"]),
            upstream_url=m["upstream_url"],
        )
        for m in data["models"]
    ]
    if {m.role for m in models} != {"det", "rec", "cls"}:
        raise ModelFetchError("the manifest must list exactly one det, rec and cls model")
    return data, models


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_archive(path: Path, entry: ModelEntry) -> None:
    size = path.stat().st_size
    if size != entry.size:
        raise ModelFetchError(
            f"{entry.archive}: size {size} does not match the manifest ({entry.size}); "
            "incomplete or wrong file"
        )
    actual = sha256_of(path)
    if actual != entry.sha256:
        raise ModelFetchError(
            f"{entry.archive}: sha256 {actual} does not match the manifest ({entry.sha256})"
        )


def safe_extract(archive: Path, entry: ModelEntry, dest: Path) -> Path:
    """Extract only the three model files of a verified archive into
    dest/<role>/<name>/, refusing anything else the archive contains."""
    target = dest / entry.role / entry.name
    seen: set[str] = set()
    with tarfile.open(archive) as tar:
        members = []
        for member in tar.getmembers():
            path = PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts:
                raise ModelFetchError(f"{entry.archive}: unsafe member path {member.name!r}")
            if member.isdir():
                continue
            if path.name.startswith("._"):
                # macOS AppleDouble metadata (the upstream cls archive ships
                # some): never extracted, never read.
                continue
            if not member.isfile():
                raise ModelFetchError(f"{entry.archive}: non-regular member {member.name!r}")
            if len(path.parts) != 2 or path.parts[0] != entry.name or path.name not in MODEL_FILES:
                raise ModelFetchError(f"{entry.archive}: unexpected member {member.name!r}")
            if path.name in seen:
                raise ModelFetchError(f"{entry.archive}: duplicate member {member.name!r}")
            seen.add(path.name)
            members.append(member)
        missing = MODEL_FILES - seen
        if missing:
            raise ModelFetchError(f"{entry.archive}: missing {sorted(missing)}")

        staging = Path(tempfile.mkdtemp(prefix=f".{entry.name}-", dir=dest))
        try:
            for member in members:
                source = tar.extractfile(member)
                assert source is not None
                with source, (staging / PurePosixPath(member.name).name).open("wb") as out:
                    shutil.copyfileobj(source, out, CHUNK)
            if target.exists():
                shutil.rmtree(target)
            target.parent.mkdir(parents=True, exist_ok=True)
            staging.rename(target)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
    return target


def mirror_url(manifest: dict, entry: ModelEntry) -> str:
    mirror = manifest["mirror"]
    return f"https://github.com/{mirror['repository']}/releases/download/{mirror['tag']}/{entry.archive}"


def _github_asset_api_url(manifest: dict, entry: ModelEntry, token: str) -> str:
    """Private repositories serve release assets only through the API."""
    mirror = manifest["mirror"]
    request = urllib.request.Request(
        f"https://api.github.com/repos/{mirror['repository']}/releases/tags/{mirror['tag']}",
        headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        release = json.load(response)
    for asset in release.get("assets", []):
        if asset.get("name") == entry.archive:
            return asset["url"]
    raise ModelFetchError(f"release {mirror['tag']} has no asset {entry.archive}")


def download(url: str, destination: Path, headers: dict[str, str] | None = None) -> None:
    request = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(request, timeout=300) as response, destination.open("wb") as out:
        shutil.copyfileobj(response, out, CHUNK)


def obtain_archive(
    manifest: dict,
    entry: ModelEntry,
    archive_dir: Path,
    source: str,
    token: str | None,
    downloader=download,
) -> Path:
    cached = archive_dir / entry.archive
    if cached.exists():
        try:
            verify_archive(cached, entry)
        except ModelFetchError as exc:
            raise ModelFetchError(
                f"cached archive is not the pinned file ({exc}); delete {cached} to refetch"
            ) from exc
        return cached

    archive_dir.mkdir(parents=True, exist_ok=True)
    partial = cached.with_suffix(cached.suffix + ".partial")
    try:
        if source == "upstream":
            downloader(entry.upstream_url, partial)
        elif token:
            api_url = _github_asset_api_url(manifest, entry, token)
            downloader(
                api_url,
                partial,
                {"Authorization": f"Bearer {token}", "Accept": "application/octet-stream"},
            )
        else:
            downloader(mirror_url(manifest, entry), partial)
        verify_archive(partial, entry)
    except ModelFetchError:
        partial.unlink(missing_ok=True)
        raise
    except Exception as exc:
        partial.unlink(missing_ok=True)
        raise ModelFetchError(
            f"{entry.archive}: download from {source} failed: {type(exc).__name__}: {exc}"
        ) from exc
    partial.replace(cached)
    return cached


def install(
    manifest_path: Path,
    dest: Path,
    archive_dir: Path,
    source: str = "mirror",
    token: str | None = None,
    downloader=download,
) -> dict[str, Path]:
    manifest, models = load_manifest(manifest_path)
    dest.mkdir(parents=True, exist_ok=True)
    installed: dict[str, Path] = {}
    for entry in models:
        archive = obtain_archive(manifest, entry, archive_dir, source, token, downloader)
        installed[entry.role] = safe_extract(archive, entry, dest)
    record = {
        "manifest_sha256": sha256_of(manifest_path),
        "models": {e.role: {"name": e.name, "sha256": e.sha256} for e in models},
    }
    (dest / "installed.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return installed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dest", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--archive-dir", type=Path, default=DEFAULT_MANIFEST.parent / "archives")
    parser.add_argument("--source", choices=["mirror", "upstream"], default="mirror")
    parser.add_argument(
        "--token-file", type=Path, help="file holding a GitHub token (BuildKit secret)"
    )
    args = parser.parse_args(argv)

    token = os.environ.get("GITHUB_TOKEN") or None
    if args.token_file and args.token_file.exists():
        token = args.token_file.read_text(encoding="utf-8").strip() or token
    try:
        installed = install(args.manifest, args.dest, args.archive_dir, args.source, token)
    except ModelFetchError as exc:
        print(f"fetch_ocr_models: {exc}", file=sys.stderr)
        return 1
    for role, path in sorted(installed.items()):
        print(f"{role}: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
