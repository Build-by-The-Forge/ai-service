"""Dependency declarations agree across installation paths (Phase 2 Step 2).

requirements*.txt are the canonical source (the Docker image, CI and local
installs use them); pyproject.toml must not drift from them.
"""

import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REQUIREMENT_FILES = ["requirements.txt", "requirements-ocr.txt", "requirements-ocr-nodeps.txt"]
OPENCV_DISTRIBUTIONS = {
    "opencv-python",
    "opencv-python-headless",
    "opencv-contrib-python",
    "opencv-contrib-python-headless",
}


def normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def parse_pin(line: str) -> tuple[str, str] | None:
    line = line.split("#", 1)[0].strip()
    if not line or line.startswith("-"):
        return None
    match = re.fullmatch(r"([A-Za-z0-9_.\-]+)(\[[^\]]*\])?\s*==\s*([^\s;]+)", line)
    assert match, f"every requirement must be an exact pin: {line!r}"
    return normalize(match.group(1)), match.group(3)


def requirement_pins() -> dict[str, list[tuple[str, str]]]:
    pins: dict[str, list[tuple[str, str]]] = {}
    for file in REQUIREMENT_FILES:
        for line in (ROOT / file).read_text(encoding="utf-8").splitlines():
            pin = parse_pin(line)
            if pin:
                pins.setdefault(pin[0], []).append((file, pin[1]))
    return pins


def pyproject_pins() -> list[tuple[str, str, str]]:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    groups = {"dependencies": project["dependencies"], **project.get("optional-dependencies", {})}
    return [
        (group, *pin)
        for group, deps in groups.items()
        for pin in (parse_pin(dep) for dep in deps)
        if pin
    ]


def test_only_the_headless_opencv_is_declared_anywhere():
    declared = {name for name in requirement_pins() if name in OPENCV_DISTRIBUTIONS}
    declared |= {name for _, name, _ in pyproject_pins() if name in OPENCV_DISTRIBUTIONS}
    assert declared == {"opencv-python-headless"}, declared


def test_opencv_pin_satisfies_paddleocr():
    # paddleocr 2.7 declares opencv-contrib-python<=4.6.0.66.
    versions = {v for _, v in requirement_pins()["opencv-python-headless"]}
    versions |= {v for _, n, v in pyproject_pins() if n == "opencv-python-headless"}
    assert versions == {"4.6.0.66"}
