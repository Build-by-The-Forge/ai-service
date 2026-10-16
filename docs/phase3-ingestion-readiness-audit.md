# Phase 3 readiness audit — document ingestion to retrieval (AI-service)

**Date:** 2026-10-16. **Baseline:** ai-service `87a0d9e` (main). The backend half is in build-backend `docs/phase3-ingestion-readiness-audit.md` (baseline `a531c10`), and the plan built on both is in `docs/phase3-ingestion-plan.md`.
**Scope:** inspection only; no code changed. Line references are to `87a0d9e`.
**Purpose:** before Phase 3 builds anything, separate what really works from what is stubbed, measured against the Phase 2 audit (`docs/phase2-ocr-document-audit.md`, 2026-10-10) and the four target stages: ingestion, cleaning and understanding, durable indexing and retrieval, and end-to-end integration.

## 1. Summary

- **The service still cannot ingest a document.** `ContentIngestionTaskHandler.handle()` fails every task with `INGESTION_PIPELINE_UNAVAILABLE` (`app/services/tasks/handlers.py:22-40`), as Phase 1 intended.
- **The plumbing around the pipeline is real and tested; the pipeline itself is missing.** Signed dispatch, idempotency, the Redis Streams worker, stage reporting, signed callbacks and task status all work. The worker already refuses to report an ingestion as successful without an `indexed` stage report (`app/services/tasks/worker.py:272`).
- **The first blocker is storage, and it is shared with the backend.** Nothing in either repository stores or fetches a document's bytes:
  - the backend accepts any non-empty `storage_uri` from its client (backend audit);
  - nothing here reads `storage_uri` (`handlers.py:33-35`);
  - `DocumentStoragePort.write_document()` builds a URI and discards the content (`app/core/documents/intake.py:40-50`).
- **Retrieval cannot see ingested content even once ingestion exists.** The retrieval engine is a per-process singleton over an in-memory store (`app/services/rag/retrieval.py:34-37, 184-188`). The worker (`ai-worker`) and the API (`ai-service`) are separate processes, so an index built by one is invisible to the other. A durable, shared index is a prerequisite, not an optimisation.
- **Phase 2 Steps 1 and 2 are done; Steps 3 to 7 have not started.** Five Phase 2 findings are still open: F2, F3, F4, F8 and F9 (section 3).

## 2. What changed since the Phase 2 audit

| Phase 2 step | State | Evidence |
|---|---|---|
| 1. Make failure observable (F1, F5) | **Done** | `PaddleOCRProvider.process()` raises `OCRNotImplementedError`, and its health check reports `unavailable` (`app/core/documents/ocr.py:110-127`). `OpenVINOService` refuses (`app/services/ocr/openvino.py:16-20`). `PgVectorStore` raises on every operation (`app/services/rag/vector_store.py:119-148`). The PaddleOCR import and model-load failures are kept and typed (`app/services/ocr/paddle.py:15-46`). No fake `openvino` module is planted. Commits `c64c5cf`, `84502e4`, `e33bb1d`, `83f53c0`, `c1a67ea`, `c69220c`. |
| 2. Pin one environment (F6, F7) | **Done** | PyMuPDF 1.28.0 and one OpenCV everywhere. Verified OCR models are baked into the image from the checksummed `ocr-models-2026.10.0` release, and real OCR runs in the built image with networking disabled. R4 (licences, evaluation material) is closed. Commits `c3a94e7` to `fd38af2`, then the image security work. |
| 3. Real page classification (F3) | Not started | Section 3 |
| 4. OCR on one interface, off the event loop (F4, F8) | Not started | Section 3 |
| 5. Normalisation that keeps meaning (F2) | Not started | Section 3 |
| 6. Document understanding | Not started | — |
| 7. Durable indexing | Not started | Section 4, Stage 3 |

## 3. Phase 2 findings still open

| # | Finding | Evidence at `87a0d9e` |
|---|---|---|
| F2 | `TextCleaner` deletes every character outside `\w \s . , ; : ( ) -`, including `+ = / % ' " ? !` | `app/services/ocr/cleaner.py:8` |
| F3 | `DocumentAnalyzer` never opens the file. Page types come from caller-supplied objects, cast across types | `app/core/documents/analysis.py:63-65` |
| F4 | Three OCR interfaces: `services/ocr/base.OCRProvider`, `core/documents/ocr.OCRProvider` and `core/documents/ports.OCRPort`. The working engine implements only the first | `app/services/ocr/base.py:4`, `app/core/documents/ocr.py:64`, `app/core/documents/ports.py:8` |
| F8 | The synchronous `self.model.ocr()` (seconds per page) runs inside an `async` method and blocks the worker's event loop | `app/services/ocr/paddle.py:48, 68` |
| F9 | Stubs: `ImageParser.parse()` returns `""`, `RAGPipeline.run()` returns `{"result": "placeholder"}` | `app/services/ocr/image_parser.py:1-3`, `app/rag/pipeline.py:1-3` |

## 4. Stage by stage

### Stage 1 — Ingestion: asset retrieval, validation, classification, extraction

| Component | State | Evidence |
|---|---|---|
| Asset retrieval from `storage_uri` | **Missing.** Nothing reads it, and there is no storage client or storage setting | `handlers.py:33-35`; `app/core/config.py` has no storage settings |
| Integrity check of the fetched bytes | **Missing.** `content_sha256` arrives with every dispatch (`app/schemas/ingest.py:37`), but no bytes are fetched to check it against | — |
| `DocumentIntakeService` validation (MIME allow-list, size limit, SHA-256, safe names, tenant from the verified scope) | **Works** on bytes it is given; not on the ingestion path | `intake.py:53-170`; unit tests |
| `DocumentStoragePort.write_document()` | **Writes nothing**: returns `tenants/{tenant}/documents/{id}/{name}` and drops `content`. Not used by ingestion | `intake.py:40-50` |
| Page classification | **Absent** (F3) | Section 3 |
| Native PDF text | **Works, per page**: `PDFParser.extract_pages()` with PyMuPDF, which is in the image | `app/services/ocr/pdf_parser.py:19-33` |
| OCR | **Works** (PaddleOCR 2.7, en PP-OCRv3), verified offline in CI on the built image, with validated input images. Still blocks the event loop (F8), behind one of three interfaces (F4) | `paddle.py`; `scripts/test_offline_ocr_image.sh` |
| OCR output validation (confidence floors, character classes, empty pages) | **Absent at runtime.** `core/documents/metrics.py` is used only by the benchmark script | `scripts/smoke_test_benchmark.py` |
| Stage evidence | **Ready**: `StageReporter` delivers `extracted`, `chunked` and `indexed` callbacks, and the worker blocks success without `indexed` | `app/services/tasks/ingestion_outcomes.py:59`; `worker.py:143, 190, 272` |
| Failure classes | **Ready**: transient and permanent ingestion errors | `ingestion_outcomes.py:33-55` |

### Stage 2 — Cleaning and document understanding

| Component | State | Evidence |
|---|---|---|
| Normalisation | **Destructive** (F2) | Section 3 |
| Chunking | **Present, too coarse for provenance.** Fixed 150-word windows with a 30-word overlap over one flat string; one `provenance` object for the whole document, so a chunk cannot be traced to its page | `app/services/rag/chunker.py:13, 28-79` |
| Layout, reading order, tables, equations | **Absent** | — |

### Stage 3 — Durable indexing and retrieval

| Component | State | Evidence |
|---|---|---|
| Embeddings | **Mock by default**: SHA-256 hash vectors with no semantic meaning. `OpenAIEmbeddingAdapter` (`text-embedding-3-small`, 1536 dimensions) exists, but nothing selects it and there is no setting for it | `app/services/rag/embedder.py:46-90`; `retrieval.py:34` |
| Vector store | **In-memory by default**, per process, lost on restart. `PgVectorStore` refuses every call | `vector_store.py:36-116, 119-148`; `retrieval.py:35` |
| pgvector | **Not available anywhere.** Every Postgres is plain `postgres:17`, and the only AI-owned migrations are the two memory tables | `.github/workflows/backend.yml:75`; `docker-compose.yml:16`; `alembic/versions/` |
| Lexical search | In-memory BM25, per process | `app/services/rag/lexical.py`; `retrieval.py:36` |
| Retrieval engine | **Per-process singleton**; worker and API never share an index | `retrieval.py:184-188` |
| Tenant scoping | **Enforced**: storage and queries require the authoritative tenant | `vector_store.py:57, 75, 141` |
| Re-indexing | `delete_document()` on the port; no versioned re-index | `vector_store.py:28, 105-113` |
| Chat context | Retrieval feeds chat (`app/services/ai/orchestrator.py:29-31`), over the empty per-process store | — |
| Retrieval-quality evaluation | **Material available, no harness.** Seven documents with ground truth in `data/ocr_evaluation` (cleared for repository use under R4). The scripts measure OCR, not retrieval | `data/ocr_evaluation/ground_truth/` |

### Stage 4 — End-to-end integration (AI-service side)

| Component | State | Evidence |
|---|---|---|
| Signed dispatch | **Works**: idempotent and fail-closed (no dispatch without a shared idempotency decision) | `app/api/v1/ingest.py:44-77` |
| Task transport | **Works**: Redis Streams `stream:ai_tasks` with a dead-letter stream | `app/services/tasks/transport.py:17-48` |
| Task records for reconciliation | **Works**: stored in Redis for 14 days | `app/services/tasks/task_records.py:26` |
| Signed callbacks | **Works**: proven by the boundary gate (C18, C22 on the failure outcome, C23) | build-backend `boundary/test_ingestion_boundary.py` |

## 5. Dependency order this implies

1. **Storage and fetch**, together with the backend. Nothing else can be exercised on real documents without it.
2. **Extraction**: classification, native text, OCR on one interface off the event loop, and output validation.
3. **Normalisation and page-aware chunking.**
4. **A durable shared index**: an embedding provider, pgvector, and Postgres full-text search.
5. **Retrieval over that index**, with a quality harness.
6. **End-to-end automation** (backend) and C22 upgraded to `ready`.

The plan, the decisions it needs, and the acceptance gates are in `docs/phase3-ingestion-plan.md`.

## 6. Not changed by this audit

No source, test, dependency or configuration was modified.
