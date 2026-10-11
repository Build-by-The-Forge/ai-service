"""Offline OCR check of a production image, with runtime dependencies only.

    python scripts/check_offline_ocr.py

Run inside the image (scripts/test_offline_ocr_image.sh does it with
--network none). Requires OCR_MODEL_DIR. Name resolution and connections to
anything but loopback are refused for the whole run, so a hidden model
download fails the check. Exit status 0 only if:
  - the verified models load,
  - real OCR reads back a rendered page exactly (confidence >= 0.8),
  - a blank page yields no text, and
  - a download attempt is refused by the guard.

Needs no pytest: the production image ships no test tooling.
"""

from __future__ import annotations

import asyncio
import os
import socket
import sys
import tempfile
from pathlib import Path

LINES = ["Photosynthesis converts light energy", "Question 3. What is osmosis?"]
LOOPBACK = {"127.0.0.1", "::1", "localhost"}


def guard_network() -> None:
    real_getaddrinfo = socket.getaddrinfo
    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex

    def host(address) -> str:
        return str(address[0] if isinstance(address, tuple) else address)

    def refuse(what: str):
        raise OSError(f"network access attempted during OCR: {what}")

    def getaddrinfo(name, *args, **kwargs):
        if str(name) not in LOOPBACK:
            refuse(f"resolve {name}")
        return real_getaddrinfo(name, *args, **kwargs)

    def connect(self, address):
        if host(address) not in LOOPBACK:
            refuse(f"connect {address}")
        return real_connect(self, address)

    def connect_ex(self, address):
        if host(address) not in LOOPBACK:
            refuse(f"connect {address}")
        return real_connect_ex(self, address)

    socket.getaddrinfo = getaddrinfo  # type: ignore[assignment]
    socket.socket.connect = connect  # type: ignore[method-assign]
    socket.socket.connect_ex = connect_ex  # type: ignore[method-assign]


def render(path: Path, lines: list[str]) -> None:
    from PIL import Image, ImageDraw, ImageFont

    font = None
    for name in ("arial.ttf", "DejaVuSans.ttf", "LiberationSans-Regular.ttf"):
        try:
            font = ImageFont.truetype(name, 34)
            break
        except OSError:
            continue
    font = font or ImageFont.load_default(size=34)
    image = Image.new("RGB", (900, 40 + 62 * max(len(lines), 1)), "white")
    draw = ImageDraw.Draw(image)
    for i, line in enumerate(lines):
        draw.text((20, 20 + i * 62), line, fill="black", font=font)
    image.save(path)


def main() -> int:
    if not os.environ.get("OCR_MODEL_DIR"):
        print("FAIL: OCR_MODEL_DIR is not set")
        return 1
    guard_network()

    import urllib.request

    try:
        urllib.request.urlopen("https://paddleocr.bj.bcebos.com/", timeout=5)
        print("FAIL: the network guard let a download through")
        return 1
    except OSError as exc:
        if "network access attempted during OCR" not in str(exc):
            print(f"FAIL: unexpected error from the network guard: {exc}")
            return 1

    from app.services.ocr.paddle import PaddleOCRService

    service = PaddleOCRService()
    with tempfile.TemporaryDirectory() as tmp:
        page, blank = Path(tmp) / "page.png", Path(tmp) / "blank.png"
        render(page, LINES)
        render(blank, [])
        lines = asyncio.run(service.extract_text_from_image(str(page)))
        empty = asyncio.run(service.extract_text_from_image(str(blank)))

    texts = [line["text"].strip() for line in lines]
    failures = []
    if [t.lower() for t in texts] != [t.lower() for t in LINES]:
        failures.append(f"read {texts!r}, expected {LINES!r}")
    if not all(line["confidence"] >= 0.8 for line in lines):
        failures.append(f"low confidence: {[round(line['confidence'], 3) for line in lines]}")
    if empty != []:
        failures.append(f"blank page produced text: {empty!r}")
    for failure in failures:
        print(f"FAIL: {failure}")
    if failures:
        return 1
    print(f"OK: offline OCR read {len(texts)} lines from {os.environ['OCR_MODEL_DIR']}: {texts}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
