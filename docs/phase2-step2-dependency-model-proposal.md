# Phase 2 Step 2 — Dependency consolidation and OCR model packaging: proposal

**Status:** proposal for review; nothing implemented. **Baseline:** ai-service `9c8bd2a` (Step 1 plus clean-up, unpushed).
**Inputs:** inspection of `pyproject.toml`, `requirements*.txt`, `Dockerfile`, the local virtualenv, the built image `build-m3-ai-service`, the PaddleOCR 2.7 source, the committed model files, and the upstream model archives (downloaded to a scratch directory only).

## 1. Findings

### 1.1 Three dependency sets that disagree

| Package | `pyproject.toml` | Docker image (`requirements*.txt`) | Local venv |
|---|---|---|---|
| PyMuPDF | `1.20.2` (no CPython 3.12 wheel, per `requirements-ocr.txt`) | **absent**, so `PDFParser` cannot work in the image | `1.28.0` |
| OpenVINO | `2024.6.0` | **absent** | `2026.3.1` |
| OpenCV | `opencv-python==4.10.0.84` (needs `libGL`, missing from the slim image) | `opencv-python-headless==4.6.0.66` | **both** `opencv-contrib-python 4.10.0.84` and `opencv-python-headless 4.6.0.66` |
| paddleocr / paddlepaddle | 2.7.0 / 2.6.2 | same (paddleocr installed `--no-deps`) | same |

- Both OpenCV distributions install into the same `cv2` package, so the venv's `cv2` is a file-level mix: it reports 4.6.0 from the contrib package's location. A mixed native install like this is a plausible cause of the intermittent OCR import failure (audit §4), though not a proven one.
- `paddleocr 2.7` declares `opencv-contrib-python<=4.6.0.66`.
- In the image `pip check` reports missing declared dependencies. That is deliberate: `--no-deps`, with every runtime import verified (`requirements-ocr-nodeps.txt`).
- The image is Python 3.12.14 and the venv is 3.12.10.

### 1.2 Models actually required

PaddleOCR 2.7 with `lang="en"` and `ocr_version="PP-OCRv3"` (our `PaddleOCRService`):

| Model | Role | Upstream archive (paddleocr's built-in URL) | Archive SHA-256 | Size |
|---|---|---|---|---|
| `en_PP-OCRv3_det_infer` | text detection | `https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_det_infer.tar` | `e740eb522521827cc83da3ba33e1c70ccfd9b3cf7d9a24a420ce4ca5ebb8794f` | 4.0 MB |
| `en_PP-OCRv3_rec_infer` | English recognition | `…/PP-OCRv3/english/en_PP-OCRv3_rec_infer.tar` | `94402088520ae3938c51eb8782da1cd7c1bc46d5531766de99f77ee9c2f23343` | 10.0 MB |
| `ch_ppocr_mobile_v2.0_cls_infer` | text-angle classifier | `…/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar` | `507352585040d035da3b1e6374694ad679a850acb0a36a8d0d47984176357717` | 2.2 MB |

- **Total about 16 MB**, against a current image of **3.24 GB**: about 0.5% growth.
- The recognition character dictionary (`en_dict.txt`) ships inside the paddleocr package.
- The extracted `inference.pdiparams` hashes match the developer's existing `~/.paddleocr` cache exactly.
- The angle classifier is **downloaded at startup but never used**: `use_angle_cls` defaults to `False`, so our `ocr(..., cls=True)` only logs a warning, and rotated scans are not corrected. Enabling it changes OCR behaviour and belongs to Step 4. Packaging still has to include it, because paddleocr 2.7 fetches it whenever it is missing.
- **Source reliability:** all three downloads succeeded (HTTP 200), but slowly. The recogniser took 72 s and the classifier 33 s (about 130 KB/s), which argues against downloading from Baidu object storage on every build.

### 1.3 The committed model files are development assets, not deployment artifacts

About 52 MB of plain-git binaries (no LFS):
- `en_PP-OCRv3_det_infer/` and its `.tar`: **byte-identical to upstream** (the archive hash matches).
- `ch_PP-OCRv3_rec_infer/` and its `.tar`: the **Chinese** recogniser, which the working path does not use.
- The `.onnx` files and the `models/ocr/openvino_models/*.xml|bin` IR files: conversions made for the AI-2B.4.3 benchmark (`scripts/benchmark_ocr_baseline.py`, `scripts/smoke_test_benchmark.py`, `docs/AI-2B-4-3-*`). Runtime does not use them.

The deployment path needs the English recogniser, which is not committed at all.

A separate finding: a third-party exam PDF (`data/ocr_evaluation/pdfs/MTS 102 … BY ROTIMI.pdf`, 4.7 MB) is committed as evaluation data. Its redistribution rights should be confirmed.

### 1.4 Offline behaviour today

`docker run --network none build-m3-ai-service` → `PaddleOCRService()` tries `https://paddleocr.bj.bcebos.com/…/en_PP-OCRv3_det_infer.tar` and fails with `ConnectionError` (name resolution). **Production OCR currently needs the internet on first use.**

Explicit model directories make this avoidable. Given `det_model_dir`, `rec_model_dir` and `cls_model_dir` that each contain `inference.pdmodel`, `inference.pdiparams` and `inference.pdiparams.info`, paddleocr's `maybe_download` performs no network access (`ppocr/utils/network.py`).

### 1.5 Licensing

- `paddleocr` 2.7.0 is Apache License 2.0 (package metadata). The PP-OCR inference models are published by the same project as part of its model zoo, under the repository's Apache-2.0 licence; redistribution inside our image is permitted with attribution and the licence notice.
- **To confirm before release:** that model-zoo terms have not changed for these exact archives, and that the NOTICE/attribution text ships in the image.
- `paddlepaddle` is Apache-2.0.

## 2. Proposal

### 2.1 Dependencies: one set, one source of truth

1. **`requirements*.txt` are the source of truth** for every runtime environment (the Docker image, CI and local). `pyproject.toml` runtime dependencies are reduced to match them exactly, and a unit test fails the build if the two disagree.
2. **OpenCV:** `opencv-python-headless==4.6.0.66` only. It satisfies paddleocr's `<=4.6.0.66` cap and needs no `libGL`. Remove `opencv-python` from `pyproject.toml`. Local venvs are rebuilt (documented); the mixed `cv2` disappears.
3. **PyMuPDF:** one version with CPython 3.12 wheels on Linux and Windows. Candidate `1.28.0`, already working in the venv. Add it to the image, so `PDFParser` works where it ships.
4. **OpenVINO:** nothing at runtime uses it; the runtime adapter has no inference (audit F1) and only the benchmark scripts need it. Move it out of the runtime set into an optional `ocr-benchmark` extra.
5. A check in the image build runs `python -c "import paddleocr, fitz, cv2"` and prints the versions, so a broken native stack fails the build rather than the first request.

### 2.2 Models: a pinned, checksummed manifest, fetched once at build

1. **`models/ocr/manifest.json`** (committed, small). For each of the three models it records name, role, version, source URL, archive SHA-256, size, licence and attribution. This is the only place model versions live.
2. **`scripts/fetch_ocr_models.py`.** It downloads each archive, **verifies its SHA-256 before extracting**, extracts into a target directory, and refuses on any mismatch, missing member or path traversal in the archive. A failed verification fails the image build.
3. **Image build.** A dedicated Docker stage runs the script into `/opt/build-ai/models/paddleocr/{det,rec,cls}`, read-only for the app user. **No download at runtime.**
4. **Application.** `PaddleOCRService` takes explicit `det_model_dir`, `rec_model_dir` and `cls_model_dir` from a new `OCR_MODEL_DIR` setting.
   - If the directories or files are missing, the service raises `OCRDependencyUnavailableError` (Step 1) naming the missing file. It never falls back to a download: in staging/production, `OCR_MODEL_DIR` is required.
   - Development may keep paddleocr's cache behaviour behind an explicit flag.
5. **Source of the archives (decision R2 below).** Recommendation: mirror the three verified archives to a BUILD-controlled location (a GitHub Release asset on a `models` tag, or GHCR as an OCI artifact) and point the manifest there, keeping the upstream URL as provenance. The checksum is the same either way; the mirror removes the slow and occasionally unreachable Baidu dependency from every build.
6. **CI** builds the image once (which fetches and verifies the models), runs the offline OCR test below against that exact image, and publishes it. Deployments pull that image; they never rebuild it.

### 2.3 Repository clean-up (separate commit, decision R3)

Remove the committed model binaries (`*_infer/`, `*.tar`, `*.onnx`, `models/ocr/openvino_models/`) from the working tree. The benchmark scripts obtain models through the same manifest, adding the Chinese recogniser and the ONNX/IR conversion as benchmark-only entries if still wanted. Git history keeps the old files. **No history rewrite is proposed** (it would change every commit id the team has pushed).

### 2.4 Tests that prove it

| Test | Proves |
|---|---|
| Manifest unit test: a corrupted archive, a wrong size, a traversal member, a missing file | `fetch_ocr_models.py` refuses all of them and never extracts an unverified archive |
| `PaddleOCRService` with an empty or partial `OCR_MODEL_DIR` | `OCRDependencyUnavailableError` names the missing file; no network call is attempted (sockets blocked in the test) |
| Dependency-agreement unit test | `pyproject.toml` runtime pins equal `requirements*.txt` |
| **Offline container test** (`docker run --network none`) | The built image initialises OCR from its packaged models and reads a rendered page, using the existing `test_real_paddle_ocr.py` inside the container. Added as a phase of the boundary gate script |
| Existing `test_real_paddle_ocr.py` locally | Real OCR still works with the explicit model directories |

### 2.5 Commit boundaries

Each commit is independently testable, and none touches OCR routing, preprocessing, text cleaning or ingestion:

1. Dependency agreement test, plus the OpenCV consolidation.
2. PyMuPDF in the image (one version everywhere).
3. OpenVINO moved to the `ocr-benchmark` extra.
4. Model manifest and the verifying fetch script, with its tests.
5. `OCR_MODEL_DIR` and explicit model directories in `PaddleOCRService`, with no download fallback in strict environments.
6. Docker model stage, plus the import and version build check.
7. The offline container OCR test in the gate.
8. Repository clean-up of the committed model binaries (only if R3 is approved).

## 3. Decisions needed

| # | Decision | Recommendation |
|---|---|---|
| R1 | Approve the single dependency set (headless OpenCV 4.6.0.66, PyMuPDF 1.28.0, OpenVINO as an optional benchmark extra) | Approve |
| R2 | Archive source: upstream Baidu URL, or a BUILD-controlled mirror with the same SHA-256 | Mirror (GitHub Release asset or GHCR), with upstream recorded as provenance |
| R3 | Remove the ~52 MB of committed model binaries from the working tree (history kept) | Approve; benchmarks fetch through the manifest |
| R4 | Confirm the model-zoo licence and attribution text for release, and the redistribution status of the evaluation PDF | Confirm before the first production image |

Out of scope for Step 2, recorded for Step 4: enabling the angle classifier (`use_angle_cls`), and moving OCR off the event loop.
