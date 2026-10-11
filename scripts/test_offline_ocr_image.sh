#!/usr/bin/env bash
# Offline OCR check of a built AI-service image (Phase 2 Step 2).
#
#   scripts/test_offline_ocr_image.sh <image>
#
# Runs scripts/check_offline_ocr.py inside the image with networking
# disabled (--network none). It uses runtime dependencies only (the image
# ships no test tooling), needs the verified models (OCR_MODEL_DIR), and
# guards the network itself, so any runtime model download fails it.
set -euo pipefail

image=${1:?usage: $0 <image>}

echo "image: $image"
docker run --rm --network none --entrypoint sh "$image" -c '
  set -e
  test -n "$OCR_MODEL_DIR" || { echo "OCR_MODEL_DIR is not set in the image"; exit 1; }
  test -f "$OCR_MODEL_DIR/installed.json" || { echo "no installed models in $OCR_MODEL_DIR"; exit 1; }
  echo "models: $OCR_MODEL_DIR"; cat "$OCR_MODEL_DIR/installed.json"
  python scripts/check_offline_ocr.py
'
