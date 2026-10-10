# Phase 2 — OCR and document-processing audit

**Date:** 2026-10-10. **Baseline:** ai-service `f3e2077` (Phase 1 final). **Scope:** inspection only; no code changed.
**Purpose:** establish what OCR and document processing actually work today, stage by stage, before Phase 2 builds the real ingestion pipeline.

## 1. Summary

- **Real OCR execution works locally.** PaddleOCR 2.7 read a rendered three-line test page correctly at 0.988–0.997 confidence: about 8 s to initialise and 3.8 s for the page on CPU.
- **Nothing reaches it at runtime.** No OCR or document class is wired into the ingestion pipeline: the `content.ingestion` handler still fails every task with `INGESTION_PIPELINE_UNAVAILABLE`, as Phase 1 intended.
- **Some of the code reports success it has not achieved.** Most serious:
  - `core/documents/ocr.PaddleOCRProvider.process()` returns no pages with `overall_confidence=1.0`, and its health check is always `ok`;
  - `PgVectorStore.add_chunks()` writes nothing but returns the chunk count;
  - `services/ocr/openvino.OpenVINOService` returns hard-coded text.

  These must not be wired into a pipeline as they are.
- **Text normalisation destroys academic content.** `TextCleaner` strips `+ = / % ' " ? !` and all other symbols.

## 2. Stage by stage

### Stage 1 — Preprocessing and page classification

| Component | State | Evidence |
|---|---|---|
| `core/documents/intake.DocumentIntakeService` | **Works** for intake: tenant from the verified scope, MIME allow-list, size limit, SHA-256, safe filenames, tenant-namespaced storage URI | Unit tests; M3-10 scope binding |
| `core/documents/analysis.DocumentAnalyzer` | **Does not analyse.** It never opens the file: the page classification (native text, scanned, image) comes from caller-supplied page objects, cast from `DocumentPage` to `PageAnalysis` (the code comments on the type mismatch). A PDF with no supplied pages becomes "unknown" and then "unsupported" | `analysis.py`; no PDF or image is read anywhere in the class |
| Image preprocessing (deskew, denoise, binarise, resize, orientation, blur/contrast checks) | **Absent.** `OCRConfig.quality_thresholds` (`min_blur_score`, `min_contrast`) is declared but read by nothing | grep: no consumer |
| `services/ocr/pdf_parser.PDFParser` | **Works where PyMuPDF is installed**: native text per page via `fitz` | Local venv has PyMuPDF 1.28 |
| `services/ocr/image_parser.ImageParser` | Stub: `parse()` returns `""` | Source |

### Stage 2 — OCR execution

| Component | State | Evidence |
|---|---|---|
| `services/ocr/paddle.PaddleOCRService` | **Works**: real PaddleOCR 2.7 call with normalised text/confidence/bbox | Probe below |
| | Blocks the event loop: the synchronous `model.ocr()` (seconds per page) runs inside an `async` method | Source |
| | Constructor loads the full model (8 s); `OCRService()` builds it eagerly | Probe |
| | Models come from PaddleOCR's runtime download cache (`~/.paddleocr/whl/...`), **not** from the model files committed in the repository. A container without network on first use cannot initialise OCR | Probe environment; Dockerfile has no model step |
| `core/documents/ocr.PaddleOCRProvider` | **Fake success**: `process()` returns `pages=[]`, `overall_confidence=1.0`; `health_check()` always `ok`. `normalize_output()` exists but nothing produces the raw input | Source |
| `core/documents/ocr.OCRRouter` | Correct routing logic (OCR only for scanned and image pages) over classifications nobody computes | Source, unit tests |
| `core/documents/openvino.OpenVINORuntime` | Real model validation and compilation adapter (path containment, device checks); **no inference** is implemented on top of it | Source |
| `services/ocr/openvino.OpenVINOService` | **Fake**: returns the constant text `"openvino extracted text"` | Source |
| OCR result validation (confidence floors, character-class checks, per-line rejection) | **Absent** | The probe read `H2O` as `H20` at 0.988 confidence: confidence alone does not catch recognition errors |

**Probe** (local, CPU): rendered text `Photosynthesis converts light energy / into chemical energy: 6CO2 + 6H2O / Question 3. What is osmosis?`.

```text
0.997 | Photosynthesis converts light energy
0.988 | into chemical energy: 6CO2 + 6H20        ← O read as zero
0.989 | Question 3. What is osmosis?
after TextCleaner:
  into chemical energy: 6CO2  6H20              ← "+" removed
  Question 3. What is osmosis                    ← "?" removed
```

### Stage 3 — Document understanding

Layout analysis, reading order, tables, equations, charts, and vision-model interpretation are **absent**. `OCRConfig.enable_table_extraction` and `enable_orientation_classification` are declared, but nothing reads them.

### Stage 4 — Ingestion and retrieval

| Component | State | Evidence |
|---|---|---|
| Ingestion handler (`content.ingestion`) | Intentionally fails with `INGESTION_PIPELINE_UNAVAILABLE` (Phase 1, M3-08) | `services/tasks/handlers.py` |
| `services/rag/chunker.Chunker` | Present; untested against real documents | Source |
| Embeddings | Default `MockEmbeddingAdapter`: SHA-256 hash vectors with **no semantic meaning**. `OpenAIEmbeddingAdapter` exists but is not the default | `retrieval.py` constructor |
| Vector store | Default `InMemoryVectorStore`: per process, lost on restart. **`PgVectorStore` is fake**: `add_chunks()` writes nothing and returns `len(chunks)`; `query_dense()` returns `[]`. No `document_chunks` table or pgvector migration exists | Source; AI-owned tables are only `learning_memory`, `session_memory` |
| Lexical index | In-memory BM25 | Source |
| Chat retrieval | Tenant-scoped (Phase 1), but always over an empty per-process store, so chat sees the placeholder context | Phase 1 boundary tests |
| `rag/pipeline.RAGPipeline` | Placeholder `{"result": "placeholder"}` | Source |

## 3. Cross-cutting findings

| # | Severity | Finding |
|---|---|---|
| F1 | High | Fake-success implementations (`PaddleOCRProvider.process`, `PgVectorStore`, `OpenVINOService`, health checks that are always ok). Wiring any of them into ingestion would report documents as processed and indexed when nothing happened. |
| F2 | High | `TextCleaner` deletes mathematical and punctuation symbols, and with them the meaning of equations, formulas and questions. |
| F3 | High | `DocumentAnalyzer` does not inspect documents, so the OCR routing decision rests on input nobody computes. |
| F4 | Medium | Three overlapping OCR interfaces: `services/ocr/base.OCRProvider`, `core/documents/ocr.OCRProvider` (same class name) and `core/documents/ports.OCRPort`. The working implementation implements only the oldest. |
| F5 | Medium | Import failures are swallowed. `paddle.py` (`except ImportError: PaddleOCR = None`) and `pdf_parser.py` replace the real exception with a generic "not installed" later. When `openvino` is absent, `core/documents/openvino.py` **injects an empty fake `openvino` module into `sys.modules`**, which any later importer of OpenVINO would see. The Docker image does not install OpenVINO, so this injection is live there. |
| F6 | Medium | Dependency drift: `pyproject.toml` pins PyMuPDF 1.20.2; the Docker requirements omit PyMuPDF entirely (so `PDFParser` fails in the image); the local venv has 1.28. The local venv also has **two** OpenCV distributions (`opencv-contrib-python` 4.10 and `opencv-python-headless` 4.6), which both install `cv2`. |
| F7 | Medium | Model provenance: `OCRConfig` names a `ch_PP-OCRv3_det_infer` detector, the repository commits an **en** detector and a **ch** recogniser (~29 MB at the repo root plus OpenVINO IR under `models/ocr`), and the working path uses neither: it downloads en det/rec at runtime. |
| F8 | Medium | Synchronous OCR inside `async` methods blocks the worker's event loop for seconds per page. |
| F9 | Low | `services/ocr/image_parser.ImageParser` and `rag/pipeline.RAGPipeline` are empty stubs. |

## 4. The OCR test failures (investigated, not assumed)

| Test | Finding |
|---|---|
| `test_service_stubs.py::test_image_ocr_providers_return_data` | **Root cause confirmed, test defect.** It hands the nonexistent path `image.png` to the *real* PaddleOCR engine, which raises `FileNotFoundError` (`paddleocr.py:526`). Deterministic; fails on every run. |
| `test_ocr_service.py` (two tests), intermittent | **Mechanism confirmed, trigger unresolved.** Both failed once in a full-suite run with "PaddleOCR is not installed": the `from paddleocr import PaddleOCR` raised during that run, and F5 discarded the exception. Two instrumented full runs (`-p probe_paddle`, which records the real exception) and repeated isolated runs did **not** reproduce it. The fake-OpenVINO hypothesis was tested and **disproved** in this environment: OpenVINO 2026.3.1 is genuinely installed, so the injection does not run. Unproven candidates: the virtualenv lives under OneDrive (Files On-Demand can fail DLL loads transiently), and the duplicate `cv2` distributions (F6). Next step: log the swallowed exception (F5 fix), so the next occurrence names its cause. |

## 5. Recommended Phase 2 sequence

Each step has its own tests. None reports success without producing output.

1. **Make failure observable (F1, F5).** Remove or fail-loudly every fake implementation: `PaddleOCRProvider.process` raises until implemented, `PgVectorStore` raises until it writes, and health checks report the real state. Keep and log the import exception, raising a typed `OCRUnavailableError(cause)`. Remove the fake `openvino` injection.
   Tests: missing dependency → typed error naming the cause; model load failure → typed error; a fake implementation can no longer return success.
2. **Pin one environment (F6, F7).** One PyMuPDF and one OpenCV, identical in pyproject, requirements and Docker. Bake the chosen models into the image from a declared, checksummed source (no runtime download). Decide what to do with the committed model binaries (Git LFS, an artifact store, or a build step).
3. **Stage 1, real classification (F3).** Open the PDF, measure native text per page (characters, text coverage, image coverage), and classify each page as native, scanned, mixed or image. Add preprocessing: orientation, deskew, contrast/blur checks against `quality_thresholds`.
   Tests: fixture PDFs (native, scanned, mixed) and images, each with its expected classification.
4. **Stage 2, OCR execution on one interface (F4, F8).** Converge on one provider port. Implement it with the working PaddleOCR path, run off the event loop (worker thread or process), and validate the output (confidence floor, character classes, empty-page handling).
   Tests: a real OCR run on fixture pages in CI (at least the 3-line probe, with expected text and a tolerance), plus missing-model and corrupt-image cases.
5. **Normalisation that keeps meaning (F2).** Replace `TextCleaner` with whitespace and Unicode normalisation that preserves symbols. Equation and formula handling belongs to Stage 3.
   Tests: `+ = ? % ²` and Greek letters survive.
6. **Stage 3, document understanding** (layout, reading order, tables, equations, charts). Scope it from real course material before choosing tools.
7. **Stage 4, durable indexing** — the plan's deferred work:
   - a real embedding provider;
   - an AI-owned `document_chunks` table with pgvector, behind an Alembic migration, tenant-keyed, under `build_ai_owner`;
   - a real `PgVectorStore`;
   - chunking validated on real documents;
   - the ingestion handler reports `extracted → chunked → indexed` only with real evidence (the M3-08 stage contract already demands this);
   - retrieval-backed chat over the durable index.

   Tests: the C22 boundary test upgraded from the failure outcome to `ready`, with stage evidence and a retrieval hit.

## 6. Not changed by this audit

No source, test or dependency was modified. The probe plugin and script live outside the repository.
