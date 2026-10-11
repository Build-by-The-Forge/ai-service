"""scripts/fetch_ocr_models.py: nothing is extracted unless verified (Phase 2 Step 2)."""

import hashlib
import importlib.util
import io
import json
import sys
import tarfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "fetch_ocr_models", ROOT / "scripts" / "fetch_ocr_models.py"
)
fetch = importlib.util.module_from_spec(_spec)
sys.modules["fetch_ocr_models"] = fetch
_spec.loader.exec_module(fetch)

FILES = ("inference.pdmodel", "inference.pdiparams", "inference.pdiparams.info")


def make_archive(
    path: Path, name: str, members: dict[str, bytes] | None = None, *, symlink: str | None = None
) -> None:
    members = (
        members if members is not None else {f"{name}/{f}": f"{name}:{f}".encode() for f in FILES}
    )
    with tarfile.open(path, "w") as tar:
        for member_name, data in members.items():
            info = tarfile.TarInfo(member_name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
        if symlink:
            info = tarfile.TarInfo(symlink)
            info.type = tarfile.SYMTYPE
            info.linkname = "/etc/passwd"
            tar.addfile(info)


def entry_for(role: str, name: str, archive: Path) -> dict:
    data = archive.read_bytes()
    return {
        "role": role,
        "name": name,
        "version": "test",
        "archive": archive.name,
        "sha256": hashlib.sha256(data).hexdigest(),
        "size": len(data),
        "upstream_url": f"https://upstream.example/{archive.name}",
        "license": "test",
        "attribution": "test",
    }


@pytest.fixture
def world(tmp_path):
    """Three valid archives in an 'origin' dir, a manifest, an empty cache and a destination."""
    origin = tmp_path / "origin"
    origin.mkdir()
    entries = []
    for role, name in (("det", "det_m"), ("rec", "rec_m"), ("cls", "cls_m")):
        make_archive(origin / f"{name}.tar", name)
        entries.append(entry_for(role, name, origin / f"{name}.tar"))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": 1,
                "mirror": {"type": "github-release", "repository": "o/r", "tag": "t"},
                "models": entries,
            }
        )
    )
    calls: list[str] = []

    def downloader(url, destination, headers=None):
        calls.append(url)
        destination.write_bytes((origin / url.rsplit("/", 1)[-1]).read_bytes())

    return {
        "tmp": tmp_path,
        "origin": origin,
        "manifest": manifest,
        "cache": tmp_path / "cache",
        "dest": tmp_path / "dest",
        "calls": calls,
        "downloader": downloader,
    }


def run(world, **kw):
    return fetch.install(
        world["manifest"], world["dest"], world["cache"], downloader=world["downloader"], **kw
    )


def test_verified_archives_install_the_three_model_files(world):
    installed = run(world)
    assert set(installed) == {"det", "rec", "cls"}
    for role, name in (("det", "det_m"), ("rec", "rec_m"), ("cls", "cls_m")):
        assert sorted(p.name for p in (world["dest"] / role / name).iterdir()) == sorted(FILES)
    assert world["calls"] == [
        f"https://github.com/o/r/releases/download/t/{n}.tar" for n in ("det_m", "rec_m", "cls_m")
    ]
    assert (
        json.loads((world["dest"] / "installed.json").read_text())["models"]["rec"]["name"]
        == "rec_m"
    )


def test_the_mirror_is_the_default_and_upstream_only_on_request(world):
    run(world, source="upstream")
    assert all(url.startswith("https://upstream.example/") for url in world["calls"])


def test_verified_cached_archives_are_used_without_downloading(world):
    run(world)
    world["calls"].clear()
    run(world)
    assert world["calls"] == []


def corrupt_origin(world, archive_name, data):
    (world["origin"] / archive_name).write_bytes(data)


def test_a_checksum_mismatch_is_refused_and_nothing_is_extracted(world):
    original = (world["origin"] / "rec_m.tar").read_bytes()
    corrupt_origin(
        world, "rec_m.tar", original[:-1] + bytes([original[-1] ^ 1])
    )  # same size, one bit
    with pytest.raises(fetch.ModelFetchError, match="sha256 .* does not match"):
        run(world)
    assert not (world["dest"] / "rec").exists()
    assert not (world["cache"] / "rec_m.tar").exists()


def test_an_incomplete_download_is_refused(world):
    original = (world["origin"] / "det_m.tar").read_bytes()
    corrupt_origin(world, "det_m.tar", original[: len(original) // 2])
    with pytest.raises(fetch.ModelFetchError, match="size .* incomplete"):
        run(world)
    assert not list(world["cache"].glob("*.partial"))


def test_a_corrupt_cached_archive_is_refused_not_replaced_silently(world):
    world["cache"].mkdir()
    (world["cache"] / "det_m.tar").write_bytes(b"tampered")
    with pytest.raises(fetch.ModelFetchError, match="cached archive is not the pinned file"):
        run(world)
    assert world["calls"] == []


def test_a_failed_download_is_reported_with_its_cause(world):
    def failing(url, destination, headers=None):
        destination.write_bytes(b"partial")
        raise ConnectionError("network unreachable")

    with pytest.raises(fetch.ModelFetchError, match="download from mirror failed: ConnectionError"):
        fetch.install(world["manifest"], world["dest"], world["cache"], downloader=failing)
    assert not list(world["cache"].glob("*"))


@pytest.mark.parametrize(
    "members,symlink,message",
    [
        ({"../evil": b"x"}, None, "unsafe member path"),
        ({"/abs/inference.pdmodel": b"x"}, None, "unsafe member path"),
        ({f"rec_m/{f}": b"x" for f in FILES} | {"rec_m/extra.py": b"x"}, None, "unexpected member"),
        ({"rec_m/inference.pdmodel": b"x", "rec_m/inference.pdiparams": b"x"}, None, "missing"),
        ({f"rec_m/{f}": b"x" for f in FILES}, "rec_m/link", "non-regular member"),
        ({f"other/{f}": b"x" for f in FILES}, None, "unexpected member"),
    ],
    ids=["traversal", "absolute", "extra-file", "missing-file", "symlink", "wrong-directory"],
)
def test_archives_with_unexpected_contents_are_refused(world, members, symlink, message):
    archive = world["origin"] / "rec_m.tar"
    make_archive(archive, "rec_m", members, symlink=symlink)
    manifest = json.loads(world["manifest"].read_text())
    manifest["models"][1] = entry_for("rec", "rec_m", archive)  # checksum matches the bad archive
    world["manifest"].write_text(json.dumps(manifest))
    with pytest.raises(fetch.ModelFetchError, match=message):
        run(world)
    assert not (world["dest"] / "rec" / "rec_m").exists()


def test_the_real_manifest_lists_exactly_the_three_required_models():
    manifest, models = fetch.load_manifest(ROOT / "models" / "ocr" / "manifest.json")
    assert {(m.role, m.name) for m in models} == {
        ("det", "en_PP-OCRv3_det_infer"),
        ("rec", "en_PP-OCRv3_rec_infer"),
        ("cls", "ch_ppocr_mobile_v2.0_cls_infer"),
    }
    assert all(
        len(m.sha256) == 64 and m.upstream_url.startswith("https://paddleocr.bj.bcebos.com/")
        for m in models
    )
    assert manifest["mirror"]["type"] == "github-release"


def test_macos_metadata_entries_are_skipped_never_extracted(world):
    archive = world["origin"] / "cls_m.tar"
    members = {f"cls_m/{f}": f"cls_m:{f}".encode() for f in FILES} | {
        "cls_m/._inference.pdmodel": b"appledouble"
    }
    make_archive(archive, "cls_m", members)
    manifest = json.loads(world["manifest"].read_text())
    manifest["models"][2] = entry_for("cls", "cls_m", archive)
    world["manifest"].write_text(json.dumps(manifest))
    run(world)
    installed = world["dest"] / "cls" / "cls_m"
    assert sorted(p.name for p in installed.iterdir()) == sorted(FILES)
    assert (installed / "inference.pdmodel").read_bytes() == b"cls_m:inference.pdmodel"
