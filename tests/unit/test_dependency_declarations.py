"""Dependency declarations agree across installation paths (Phase 2 Step 2).

requirements*.txt are the canonical source (the Docker image, CI and local
installs use them); pyproject.toml must not drift from them.
"""

import re
import tomllib
from pathlib import Path

from packaging.markers import Marker

ROOT = Path(__file__).resolve().parents[2]
# What the production image installs.
REQUIREMENT_FILES = ["requirements.txt", "requirements-ocr.txt", "requirements-ocr-nodeps.txt"]
# Test and quality tools: CI and development only, never the image.
DEV_FILE = "requirements-dev.txt"
DEV_TOOLS = {"pytest", "pytest-asyncio", "pytest-cov", "black", "ruff", "mypy", "pre-commit"}
# Installed on demand, never into the image.
BENCHMARK_FILE = "requirements-ocr-benchmark.txt"
OPENCV_DISTRIBUTIONS = {
    "opencv-python",
    "opencv-python-headless",
    "opencv-contrib-python",
    "opencv-contrib-python-headless",
}


def normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def parse_pin(line: str, platform: str = "linux") -> tuple[str, str] | None:
    # A line whose environment marker excludes `platform` is skipped; the
    # default is the image's platform.
    requirement, _, marker = line.split("#", 1)[0].partition(";")
    requirement = requirement.strip()
    if not requirement or requirement.startswith("-"):
        return None
    if marker.strip() and not Marker(marker).evaluate({"sys_platform": platform}):
        return None
    match = re.fullmatch(r"([A-Za-z0-9_.\-]+)(\[[^\]]*\])?\s*==\s*(\S+)", requirement)
    assert match, f"every requirement must be an exact pin: {line!r}"
    return normalize(match.group(1)), match.group(3)


def requirement_pins(
    files: list[str] | None = None, platform: str = "linux"
) -> dict[str, list[tuple[str, str]]]:
    pins: dict[str, list[tuple[str, str]]] = {}
    for file in files or REQUIREMENT_FILES:
        for line in (ROOT / file).read_text(encoding="utf-8").splitlines():
            pin = parse_pin(line, platform)
            if pin:
                pins.setdefault(pin[0], []).append((file, pin[1]))
    return pins


def pyproject_pins(platform: str = "linux") -> list[tuple[str, str, str]]:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    groups = {"dependencies": project["dependencies"], **project.get("optional-dependencies", {})}
    return [
        (group, *pin)
        for group, deps in groups.items()
        for pin in (parse_pin(dep, platform) for dep in deps)
        if pin
    ]


def test_only_the_headless_opencv_is_declared_anywhere():
    declared = {name for name in requirement_pins() if name in OPENCV_DISTRIBUTIONS}
    declared |= {name for _, name, _ in pyproject_pins() if name in OPENCV_DISTRIBUTIONS}
    assert declared == {"opencv-python-headless"}, declared


def test_opencv_pin_is_the_reviewed_release_for_numpy_1():
    # paddleocr 2.7 declares opencv<=4.6.0.66, which bundles a vulnerable
    # libwebp. 4.11.0.86 is the reviewed exception: the newest release that
    # accepts numpy 1.x (scipy 1.11; paddlepaddle 2.6 on Windows), with
    # identical OCR output. A different version needs the same review, not
    # just a new number here.
    versions = {v for _, v in requirement_pins()["opencv-python-headless"]}
    versions |= {v for _, n, v in pyproject_pins() if n == "opencv-python-headless"}
    assert versions == {"4.11.0.86"}
    assert {v for _, v in requirement_pins()["numpy"]} == {"1.26.4"}


def test_the_image_gets_the_fixed_ocr_engine_stack():
    # Reviewed against real OCR output in the image (OCR-stack security step).
    image = requirement_pins()
    assert image["paddlepaddle"] == [("requirements-ocr.txt", "3.3.1")]
    assert image["safetensors"] == [("requirements-ocr.txt", "0.8.0")]
    assert image["protobuf"] == [("requirements-ocr.txt", "5.29.6")]
    assert image["setuptools"] == [("requirements-ocr.txt", "83.0.0")]


def test_windows_development_keeps_paddle_2_6_with_its_protobuf_cap():
    # Windows App Control on the development machine blocks paddle 3's
    # native library. paddlepaddle 2.6 declares protobuf<=3.20.2 on Windows.
    windows = requirement_pins(platform="win32")
    assert windows["paddlepaddle"] == [("requirements-ocr.txt", "2.6.2")]
    assert windows["protobuf"] == [("requirements-ocr.txt", "3.20.2")]
    assert "safetensors" not in windows


def test_pyproject_follows_the_platform_split():
    for platform, version in [("linux", "3.3.1"), ("win32", "2.6.2")]:
        pins = [v for _, n, v in pyproject_pins(platform) if n == "paddlepaddle"]
        assert pins == [version], platform


def test_each_package_is_pinned_once_on_windows_too():
    duplicates = {
        n: places for n, places in requirement_pins(platform="win32").items() if len(places) > 1
    }
    assert duplicates == {}


def test_pymupdf_is_one_version_everywhere_and_installed_in_the_image():
    in_requirements = requirement_pins()["pymupdf"]
    assert [f for f, _ in in_requirements] == ["requirements-ocr.txt"]  # the image installs it
    versions = {v for _, v in in_requirements} | {
        v for _, n, v in pyproject_pins() if n == "pymupdf"
    }
    assert versions == {"1.28.0"}


def test_every_pyproject_pin_agrees_with_the_requirements_files():
    image = requirement_pins()
    benchmark = requirement_pins([BENCHMARK_FILE])
    dev = requirement_pins([DEV_FILE])
    for group, name, version in pyproject_pins():
        source = {"ocr-benchmark": benchmark, "dev": dev}.get(group, image)
        assert name in source, f"{group}: {name} is not pinned in the requirements files"
        assert {v for _, v in source[name]} == {version}, (group, name, version, source[name])


def test_each_package_is_pinned_once_across_the_image_requirements():
    duplicates = {n: places for n, places in requirement_pins().items() if len(places) > 1}
    assert duplicates == {}


def test_openvino_is_benchmark_only():
    assert "openvino" not in requirement_pins()
    assert [(g, n) for g, n, _ in pyproject_pins() if n == "openvino"] == [
        ("ocr-benchmark", "openvino")
    ]


def test_no_development_tool_ships_in_the_image():
    image = requirement_pins()
    shipped = {name for name in image if name in DEV_TOOLS or name.startswith("types-")}
    assert shipped == set(), shipped


def test_the_dev_set_builds_on_the_runtime_set():
    lines = (ROOT / DEV_FILE).read_text(encoding="utf-8").splitlines()
    assert "-r requirements.txt" in [line.strip() for line in lines]
    assert DEV_TOOLS <= set(requirement_pins([DEV_FILE]))
