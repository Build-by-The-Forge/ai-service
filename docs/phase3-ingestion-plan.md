# Phase 3 plan — document ingestion to retrieval

**Date:** 2026-10-16. **Status:** PROPOSED. Decisions D1 and D2 are needed before P3-1 starts.
**Built on:**
- ai-service `docs/phase3-ingestion-readiness-audit.md` (`87a0d9e`);
- build-backend `docs/phase3-ingestion-readiness-audit.md` (`a531c10`).

## 1. Goal and definition of done

Phase 3 is done when the M3 design's original C22 passes in the boundary gate. A tenant uploads a document and BUILD stores it. The job reaches `ready` through real `extracted`, `chunked` and `indexed` evidence. A question answered by that document retrieves a chunk from it, for that tenant only.

Every stage must produce its output before it reports success.

## 2. Rules for every milestone

- **No fake success.** A stage reports progress only after its output exists. Failures are typed as transient (retryable) or permanent.
- **Tenant from the verified context only.** Storage locations carry the tenant, and both services check it.
- **Real dependencies in integration tests.** Postgres, Redis and the object store run in CI. Doubles are only for what a unit test does not exercise.
- **Small, separate commits and PRs.** Each must keep green:
  - ai-service: both workflows, with the 80% coverage floor, real OCR in the built image, and the image audit;
  - build-backend: CI and the boundary gate.
- **Publication stays gated.** `PUBLISH_IMAGES` is not set.
- **New images, packages and models are pinned and audited.** Each gets a licence review, as R4 did.

## 3. Decisions needed

| # | Decision | Options | Recommendation |
|---|---|---|---|
| D1 | Where uploaded bytes live | (a) S3-compatible object storage: MinIO pinned by digest locally and in CI; the production provider to be chosen. (b) A volume shared by both services | **(a).** A shared volume ties both services to one host and has no per-object access control. |
| D2 | How bytes arrive | (a) The backend receives the upload, measures size and SHA-256, checks the MIME type, writes the object, and generates `storage_uri`. (b) The client uploads directly with a presigned URL | **(a) for Phase 3.** The backend then never trusts client-declared size, hash or location. Presigned uploads can follow if traffic needs them. |
| D3 | Embedding provider | (a) OpenAI `text-embedding-3-small` through the existing adapter: course documents leave BUILD for OpenAI, a new class of data sent to a provider already used for chat. (b) A local model in the worker image: no data leaves, but a larger image, CPU cost and a licence review | **Your call.** (a) is the smaller build, if sending document text to OpenAI is acceptable. Either way, the dimension is pinned in configuration and in the migration. |
| D4 | Vector and keyword storage | (a) pgvector plus Postgres full-text search in the AI database: one durable store for hybrid retrieval, replacing the in-memory BM25. (b) A separate vector database | **(a)**, if the production Postgres host supports the `vector` extension. Confirm with the host before P3-4. |
| D5 | Who moves jobs | (a) A backend worker process (same image, its own entrypoint) that dispatches queued jobs, runs due retries, and reconciles jobs stuck in `processing`. Row locks (`FOR UPDATE SKIP LOCKED`) keep several workers safe. (b) An external scheduler calling the existing endpoints | **(a).** It uses the domain rules directly and needs no extra infrastructure. |

## 4. Milestones

Each milestone is a series of small PRs, merged in order.

### P3-1 — Storage and verified fetch (backend and AI-service; needs D1, D2)

**Scope**
- **Storage service:** MinIO in `deploy/compose.m3.yml`, in local compose and in CI, pinned by digest, with one bucket.
- **Backend upload:** an upload endpoint that stores the bytes with a server-measured size and SHA-256 and a checked MIME type. It generates `storage_uri` as `s3://<bucket>/tenants/<tenant>/assets/<asset>/<sha256>`.
  - Registration with a client-supplied `storage_uri` is removed or restricted to validated internal use.
  - Storage credentials are scoped: the backend can write; the AI service can only read.
- **AI fetcher:** it accepts only the configured bucket and the verified tenant's prefix. It streams the object to a temporary file under a size limit, compares its SHA-256 with the dispatched `content_sha256`, and checks the MIME type. It always cleans up.
- **Ingestion handler:** fetches and verifies the asset, then still raises `INGESTION_PIPELINE_UNAVAILABLE`, because nothing downstream exists yet.

**Acceptance**
1. Uploaded bytes are stored once. A client-declared size or hash that differs from the measured one is rejected.
2. The AI service refuses a `storage_uri` outside the verified tenant's prefix or the configured bucket.
3. A tampered object fails the job permanently with `ASSET_INTEGRITY_MISMATCH`, and a missing object with `ASSET_NOT_FOUND`. Both are covered by boundary tests.
4. A storage outage fails the attempt as transient (retryable).
5. C22 still ends with `INGESTION_PIPELINE_UNAVAILABLE`, now after a verified fetch that is recorded in the task's metrics.

**Gates**
- Integration tests against MinIO in CI.
- New boundary cases.
- Storage credentials appear in no image layer or log; the existing token-leak check is extended to them.

### P3-2 — Extraction (AI-service; closes F3, F4, F8)

**Scope**
- **One OCR port:** keep a single interface and retire the other two. PaddleOCR runs off the event loop, in a worker thread or process.
- **Page classification:** open the PDF with PyMuPDF and measure, per page, the native text characters and the image coverage. Each page is classified native, scanned, mixed or image.
- **Text per page:**
  - native pages use PyMuPDF text;
  - scanned and image pages are rendered at a fixed DPI and OCR'd;
  - image uploads are OCR'd directly.
- **Per-page result:** page number, method, text, confidence and warnings.
- **Output validation:** a confidence floor, empty-page handling, page-count and time limits.
- **Stage report:** `extracted`, with page counts, OCR page counts, mean confidence and characters.

**Acceptance**
1. Fixture PDFs (native, scanned, mixed) and images get their expected classification.
2. Real OCR on rendered fixture pages in CI matches the expected text within a stated tolerance.
3. The worker's heartbeat keeps running during OCR, so the event loop is free.
4. A corrupt or encrypted PDF fails permanently with a typed code.
5. In the boundary gate, the job reaches `extracted`, then fails with `INGESTION_PIPELINE_UNAVAILABLE`.

**Evaluation**
- Record a CER/WER baseline per document on the seven cleared documents with ground truth (`data/ocr_evaluation`).
- Later changes may not regress it. This is a no-regression gate, not an absolute target yet.

### P3-3 — Normalisation and page-aware chunking (AI-service; closes F2)

**Scope**
- **Normalisation:** a Unicode NFC and whitespace normaliser replaces `TextCleaner`. It keeps symbols and does not apply NFKC, which would turn `²` into `2`.
- **Chunking:** chunks carry their page span and source.
- **Chunk ids:** deterministic, derived from asset, run and index.
- **Stage report:** `chunked`, with counts.

**Acceptance**
1. `+ = ? % ² ≤ π` and Greek letters survive normalisation.
2. Every chunk maps back to its pages.
3. The same input gives the same chunk ids.
4. In the boundary gate, the job reaches `chunked`.

### P3-4 — Durable index (AI-service; needs D3, D4)

**Scope**
- **Postgres image:** pgvector, pinned by digest, in CI, compose and the boundary topology.
- **Migration:** an Alembic migration for `document_chunks`: tenant, asset, run, chunk id, content, `vector(d)`, `tsvector`, page span and timestamps. The table is owned by `build_ai_owner`, with tenant-first indexes.
- **Vector store:** a real `PgVectorStore`.
- **Embedding adapter:** the provider chosen in D3, with batching, timeouts and typed transient errors.
- **Re-indexing:** replaces an asset's chunks in one transaction.
- **Stage report:** `indexed`, only after the transaction commits.

**Acceptance**
1. The migration passes `scripts/review_migration.py`.
2. The index survives a restart.
3. Tenant B never retrieves tenant A's chunks (integration test).
4. Re-indexing the same asset is idempotent.
5. An embedding outage fails the attempt as transient and leaves nothing partially indexed.
6. In the boundary gate, the job reaches `ready`.

### P3-5 — Retrieval over the durable index (AI-service; closes F9)

**Scope**
- **Hybrid retrieval:** `HybridRetrievalEngine` over pgvector and Postgres full-text search, combined with reciprocal rank fusion.
- **Default engine:** the in-memory singleton stops being the default and stays only in tests.
- **Consumers:** chat, explain and question generation read the durable index.
- **Stubs:** `RAGPipeline` and `ImageParser` are implemented or removed.

**Acceptance**
1. The API process retrieves what the worker indexed.
2. Chat context cites chunk provenance (document and pages).
3. A retrieval-quality harness on the cleared documents (questions derived from the ground truth) records a recall@5 baseline and fails on regression.

### P3-6 — Automation and C22 restored (backend and AI-service; needs D5)

**Scope**
- **Backend worker:**
  - automatic dispatch of queued jobs;
  - due retries;
  - reconciliation of jobs in `processing` past a deadline;
  - safe with several worker instances.
- **Boundary tests:**
  - C22 becomes upload → `ready`, with stage evidence and a retrieval hit;
  - C23 also covers the success path.

**Acceptance**
1. The happy path needs no manual endpoint call.
2. Stopping the AI worker mid-job leads to reconciliation and a retry, not a stuck job.
3. The boundary gate is green with C22 as designed.

## 5. Order

1. **Decide D1 and D2, then build P3-1**; everything else needs real bytes.
2. **P3-2, then P3-3.**
3. **Decide D3 and D4, then build P3-4, then P3-5.**
4. **Decide D5 and build P3-6.** Its dispatch and retry loop needs only P3-1, so it can be built in parallel with P3-2 to P3-5. C22 as designed closes it last.

**First PRs of P3-1, once D1 and D2 are decided:**
1. MinIO in compose and CI, pinned.
2. The backend storage port and upload endpoint.
3. Backend `storage_uri` generation and validation.
4. The AI fetcher with verification.
5. The handler's fetch and verify step.
6. The new boundary cases.

## 6. Out of scope

- **Document understanding:** layout, reading order, tables, equations and charts (Phase 2 Step 6). Scope it from real course material after P3-5.
- **Video.**
- **Presigned direct uploads.**
- **Publishing images:** a separate decision, gated by `PUBLISH_IMAGES`.

## 7. Risks

| Risk | Mitigation |
|---|---|
| The production Postgres host lacks pgvector | Confirm before P3-4 (D4); a separate vector database is the fallback |
| Embedding cost and data handling | D3 is an explicit owner decision; batching and limits in P3-4 |
| OCR throughput: seconds per page on CPU | Off the event loop (P3-2); page and time limits; worker concurrency sized from P3-2 measurements |
| Large or hostile files | Size and page limits, MIME checks, typed permanent failures (P3-1, P3-2) |
