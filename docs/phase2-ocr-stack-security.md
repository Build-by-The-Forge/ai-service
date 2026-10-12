# Phase 2: OCR stack security audit

Date: 2026-10-12. Branch: `feat/m3-boundary`. Scope: the Python packages in
the AI-service production image (the OCR stack in particular), malformed
input handling in OCR, the CI gate for both, and the backend image.

## Outcome

| | Before (`build-m3-ai-service`, 2026-10-11) | After (this branch) |
|---|---|---|
| Known advisories in the image (pip-audit 2.10.1, distinct ids) | 26, in 6 packages | 1, accepted with evidence (imgaug) |
| Development tools in the image | pytest, black, ruff, mypy, pre-commit, type stubs | none |
| Installed distributions | 107 | 76 |
| Malformed or oversized image input | crashed inside OpenCV, or a ~900 MB allocation | refused with `OCRExtractionError` before decoding |
| CI audit of the image's packages | none (the security job audits the dev environment, which has no OCR stack) | `scripts/audit_image_packages.py` in the `docker` job |
| Real OCR output on the evaluation corpus (797 lines) | baseline | identical text, confidences and boxes |
| Image size | 3.25 GB | 3.47 GB (paddle 3 is larger) |

## Method

- **Advisories.** `pip freeze --all` inside the built image, then
  `pip-audit --no-deps --disable-pip -r` on exactly that list. This audits
  what ships, not what a manifest resolves to on another machine.
- **Behaviour.** Every engine-level change was compared with real OCR
  over the evaluation corpus (`data/ocr_evaluation`: 17 pages, 797 lines),
  in the Linux image, with networking disabled and the corpus mounted
  read-only. Only aggregate figures were recorded (the corpus may hold
  personal data, R4). A change passes only if line text, confidences and
  bounding boxes are all identical to the baseline.
- **Image check.** `scripts/test_offline_ocr_image.sh` on each rebuilt image
  (`--network none`): verified models load, a real two-line page reads
  exactly, a blank page reads nothing, any download is refused, and
  malformed or oversized images are refused.
- **Suite.** Full test suite on fresh PostgreSQL 17 and Redis 8, with the
  80% coverage floor.

## Package changes

| Package | Before | After | Advisories cleared | Evidence |
|---|---|---|---|---|
| Pillow | 10.4.0 | 12.3.0 | 17 (PYSEC-2026-165, -2249, -2250, -2252 to -2257, -2874, -3451, -3453, -3454, -3493 to -3496) | real-OCR tests, full suite, offline image check (36f3354) |
| opencv-python-headless | 4.6.0.66 | 4.11.0.86 | 2 (PYSEC-2023-184, GHSA-jh2j-j4j9-crg3: bundled libwebp) | corpus identical; offline image check (aa8b6e9) |
| protobuf | 3.20.2 | 5.29.6 (Linux) | 2 (PYSEC-2026-1805, -1806) | corpus identical; backend changes from pure Python to the compiled `upb` (092a028) |
| setuptools | 75.6.0 | 83.0.0 | 2 (PYSEC-2025-49, PYSEC-2026-3447) | corpus identical (092a028) |
| paddlepaddle | 2.6.2 | 3.3.1 (Linux) | 2 (PYSEC-2026-1754, -1756) | corpus identical; offline image check (60f4e0a) |
| safetensors | absent | 0.8.0 (Linux) | new dependency of paddle 3.3; no advisories | |
| imgaug | 0.4.0 | 0.4.0 | none: no fixed release; accepted below | |

Notes on the choices:

- **OpenCV 4.11.0.86** is above paddleocr 2.7's declared cap
  (`<=4.6.0.66`), which `--no-deps` does not enforce. It is the newest
  release that accepts numpy 1.x; 4.12+ require numpy 2, which scipy 1.11
  in this stack is not built for. Nothing is reported from 4.8.1.78 on.
- **protobuf 5.29.6** is the smallest upgrade that clears both advisories.
  6.33.6 and 7.36.2 also passed the image check; 5.29.6 was taken as the
  smaller step.
- **paddlepaddle 3.3.1 over 3.0.0.** Both clear the advisories and gave
  identical corpus output. 3.3.1 is the current stable line, where future
  fixes will land, and it declares the setuptools dependency that paddle
  already needed at runtime. Corpus run time in one run each: 2.6.2 381 s,
  3.0.0 200 s, 3.3.1 280 s (not a benchmark; same machine, sequential).
- **setuptools and imgaug must ship.** paddle imports setuptools when it
  loads, and paddleocr 2.7 imports imgaug unconditionally. Blocking either
  stops paddleocr loading, so neither can be removed.

## Platform split: Windows development

The image (Linux) and CI run paddlepaddle 3.3.1 with protobuf 5.29.6.
Windows development installs keep paddlepaddle 2.6.2 with protobuf 3.20.2,
using environment markers in `requirements-ocr.txt` and `pyproject.toml`:

- Windows App Control on the development machine blocks paddle 3's native
  library (`libpaddle`: "An Application Control policy has blocked this
  file"). paddle 2.6.2's library still loads.
- paddlepaddle 2.6 declares `protobuf<=3.20.2` on Windows, so Windows
  keeps that too.

Consequence: local real-OCR tests on Windows exercise the older engine.
The shipped engine is verified in the image: the CI offline check and the
corpus comparison above. **Exit condition:** once Windows can load paddle 3
(the library is allowed, or development moves to WSL or Linux), remove the
markers so every platform runs one version.
`tests/unit/test_dependency_declarations.py` pins both sides of the split,
so it can't change silently.

## Accepted advisory: imgaug PYSEC-2026-356

- **Advisory:** CVE-2026-31235 / GHSA-g82g-j283-hj97. `imgaug.multicore`'s
  `BackgroundAugmenter` unpickles data from a multiprocessing queue in
  `_augment_images_worker()`, so whoever can write to that queue can run
  code. No fixed release exists (imgaug is unmaintained; 0.4.0 is the
  last release).
- **Why it ships:** paddleocr 2.7 imports imgaug at load time
  (`ppocr/data/imaug/iaa_augment.py`, training-time augmentation).
- **Exposure in this service:**
  - nothing in the image outside imgaug itself references
    `BackgroundAugmenter` or `imgaug.multicore` (searched site-packages
    and `app/`);
  - during real OCR in the image, `imgaug.multicore` is never imported,
    and no `BackgroundAugmenter` exists after the run;
  - the service never creates or exposes a multiprocessing queue to
    imgaug.
- **Decision:** accepted. Listed in `security/accepted-image-advisories.txt`.
- **Revisit when:** paddleocr is upgraded (later releases restructure
  their imports), or anything starts using imgaug's multicore API.

## Malformed and oversized image input (218c0cf, 1ee0466)

Before, input reached OpenCV inside paddleocr unchecked:
- a zero-byte file raised `cv2.error`;
- random or truncated data raised `AttributeError`;
- a 30000×30000 PNG made OpenCV attempt about 900 MB;
- `OCRConfig.max_image_pixels` (16M) existed but was never enforced.

`PaddleOCRService.extract_text_from_image` now validates first:
- only Pillow's PNG, JPEG, TIFF and WEBP decoders are allowed, and the
  content decides the format, not the file name;
- width × height is checked from the header against `max_image_pixels`
  before any pixel is decoded;
- the image is then fully decoded, so truncated data fails;
- every rejection is `OCRExtractionError` with the original error kept
  as its cause.

The offline image check proves this on the production stack. Negative
control: run against the image built before the fix, the check fails on
all five inputs.

## CI gate for the image (this commit)

The `security` job runs pip-audit on the environment from
`requirements-dev.txt`, which has no OCR stack, so the image's OCR
packages were never audited. The `docker` job now runs
`scripts/audit_image_packages.py ai-service:ci` after the offline OCR
check and before publishing:

- it audits `pip freeze --all` from the built image (pip-audit 2.10.1,
  pinned);
- it fails on any advisory not listed in
  `security/accepted-image-advisories.txt`;
- it fails on any listed advisory that no longer appears, so an
  acceptance can't outlive its reason.

Verified: on the 3.3.1 image it passes with the one accepted imgaug entry.
With a deliberately wrong list (only a paddle id) it exits 1, reporting
imgaug as unaccepted and the paddle entry as stale.

## Operational risks

- **Paddle oneDNN error.** One full corpus run on paddle 2.6.2 raised
  "could not create a primitive" inside Paddle's oneDNN backend. It did not
  recur in two reruns or with those pages alone, and the paddle 3.3.1 runs
  were clean. The cause is not established. If it appears in production,
  the request fails with `OCRExtractionError` (no fabricated text). Watch
  the OCR error rate after deployment.
- **Windows App Control on the development machine** now also blocks
  ujson's DLL, which the venv has only through modelscope (not a project
  dependency; not in the image). fastapi treats ujson as optional but
  only catches `ModuleNotFoundError`, so test collection fails locally.
  The suite was run with ujson hidden from the import system for the
  test process only. That matches the image, which has no ujson.
- **Image size** grew 0.22 GB with paddle 3.

## Backend image (`build-backend`, audited, not changed)

`build-m3-backend:latest` (built 2026-10-11 from `feat/learning-content`):

| # | Finding | Severity | Detail |
|---|---|---|---|
| B1 | Dependencies are not pinned | High | `pyproject.toml` uses lower bounds only (`fastapi>=0.115.0`, ...), so every build installs whatever is newest. The image audited clean today (69 distributions, no known advisories), but that says nothing about the next build, and builds are not reproducible. |
| B2 | Development tools ship in production | Medium | The Dockerfile runs `pip install .[dev]`: pytest, ruff, mypy, pre-commit, respx and aiosqlite are in the image. |
| B3 | The container runs as root | Medium | No `USER`; `id -un` in the image is `root`. |
| B4 | The image carries non-runtime files | Low | `COPY . .` brings `tests/`, `scripts/`, `boundary/`, `pg_smoke/`, `deploy/`, the Dockerfile, pip's `build/` leftovers and two stray files (see B6). `.dockerignore` does correctly exclude `.env*`, `deploy/m3.env`, databases and logs; no secret was found in the image. |
| B5 | A second, stale manifest | Low | `requirements.txt` (unpinned) differs from `pyproject.toml` and lists `python-jose` and `passlib`, which nothing imports. README still tells developers to install from it. |
| B6 | Stray files in the repository | Low | `tatus --short` (committed in 372fe3e) and `t -q` (untracked) are captured terminal output (a `git diff --check` and the `less` help screen). No secrets. |
| B7 | No dependency audit in backend CI | Medium | `.github/workflows/ci.yml` installs `.[dev]` and runs tests; there is no pip-audit step and no image audit. |

Recommended fix, as its own step:
1. Pin exact runtime and dev versions, using the AI-service split as the
   model.
2. Install runtime only in the image.
3. Add a non-root user.
4. Copy only `app/`, `alembic/` and `alembic.ini`.
5. Delete `requirements.txt` and the stray files.
6. Add pip-audit and an image audit to CI.

## Remaining

- The hosted CI run of these gates has not happened yet. Every result
  above was produced locally, in the same Linux image CI builds.
- The imgaug acceptance goes away when paddleocr moves past 2.7.
- Remove the Windows markers once Windows can load paddle 3.
- Backend fixes B1 to B7.
