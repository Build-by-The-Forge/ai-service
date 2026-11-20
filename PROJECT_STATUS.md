# Project Status: AI Service

**Service:** `ai-service` (Build Core). **Checked against:** `origin/main` `2c76c93` (2026-11-16, after #64), on 2026-11-20.

**What this file is:** a status statement with every claim checked in this repository. A path in backticks is where the claim was read. The delivery plan and the milestone register live in `build-backend` (`docs/BUILD_IMPLEMENTATION_PLAN.md`, `docs/milestone-register.md`); this file doesn't track milestones.

**Tests:** a local `pytest tests --collect-only` on this commit collected 1,493 tests with 23 collection errors (modules whose dependencies weren't installed on the checking machine), so 1,493 is a lower bound; the lane that owns the CI environment reports about 1,730. Collected is not passed. CI runs the suite with a coverage floor of 80% (`.github/workflows/backend.yml`, "Run the full test suite"), and CI is the authority for the count and for passing. This file claims no pass count.

## 1. What the service is

The intelligence and retrieval service behind the BUILD backend. It's called only by the backend, with signed requests (BUILD Service Signature v1, `app/core/service_signature_v1.py`), and it signs its responses (`app/core/signed_response.py`). It authenticates services, not users: `app/` has no JWT code, no RBAC, no exam runtime and no WebSocket, and `app/main.py` mounts only the routers listed below.

## 2. What exists on `main`

| Capability | Where | State |
|---|---|---|
| Service boundary: signature verification, replay protection | `app/core/middleware/service_auth.py`, `app/core/replay_guard.py` (`RedisReplayGuard`) | Built |
| Model Gateway: one execution boundary for chat, explain, question generation and the CBT generator | `app/core/models/gateway.py`, `resilience.py`, `budget.py`; design `docs/ai-01-model-gateway-design.md` | Built. Providers: `openai`, `deterministic`, `mock` (`app/core/models/providers/`). The legacy `LLMProvider` layer and the `openai` SDK are removed (#63) |
| Provider approval | `MODEL_PROVIDERS_APPROVED` and `GATEWAY_PROVIDERS` in `app/core/config.py` | Built. Empty by default, so no provider is eligible for student passages. `MODEL_PROVIDER=deterministic` is refused in strict environments |
| Gemini adapter | #65, open | **Not on main** |
| Fireworks adapter | None | **Not started**; waits for governance |
| Task queue and worker | `app/services/tasks/` (`TaskTransport`, `InMemoryTaskTransport`, `RedisStreamTransport` in `transport.py`; `TaskWorker` in `worker.py`), `app/worker.py` | Built |
| Signed dispatch and stage callbacks | `app/api/v1/ingest.py` (`POST /v1/ingest/dispatch`, `GET /v1/ingest/tasks/{external_task_id}`), `app/core/callbacks/` | Built |
| Scope assignments | `app/api/v1/scopes.py` (`POST /v1/ingest/scope-assignments`), `app/core/rag/scope.py`, migration `d4a9c1e7f203_asset_scopes.py` | Built |
| Ingestion pipeline | `ContentIngestionTaskHandler` in `app/services/tasks/handlers.py`; `app/services/ingestion/` (fetch, extraction, normalise, chunking); `app/services/ocr/` (PaddleOCR, OpenVINO, PDF and image parsers) | Built. Extraction, asset fetch and the ONNX embedder use `asyncio.to_thread` (`extraction.py`, `asset_fetcher.py`, `onnx_embedder.py`) |
| Durable vector index (pgvector) | `app/services/rag/pg_index.py`, `vector_store.py`; migration `c3e81f4a9d27_document_chunks.py`; CI enables the `vector` extension | Built. Not a stub |
| Retrieval: hybrid vector and lexical, evidence, provenance, scope filter | `app/services/rag/retrieval.py`, `lexical.py`, `evidence.py`, `recorder.py` (`sources_used`), `scopes.py`; `POST /v1/ai/retrieve` | Built |
| Tutor, explain, question generation, knowledge tree, read path | `app/api/v1/ai/{chat,explain,generate_questions,tree,read}.py`; `app/services/ai/orchestrator.py`, `app/services/generation/`, `app/services/tree/` | Built; capability gaps are in the blueprint's map |
| CBT exam generator (AI-10) | `app/services/generation/cbt_exam.py`, `cbt_ai_generator.py` (#61, #62) | Structured and grounded; **no route serves it** |
| Agent runtime and tools | `app/services/agents/executor.py` (`AgentExecutor`, bounded by `max_steps`), `app/core/agents/`, `app/core/tools/{registry,contracts,budget,validator}.py`, `app/services/tools/builtins/{content,curriculum,question}.py` | Built, **not exposed on any route** |
| Learning Memory | `app/api/v1/ai/memory.py` (`/memory/query`, `/memory/upsert`), `app/services/memory/`; gate in `app/core/learning_memory_gate.py` | Built; **gated off**: `GATE_OPEN` is `False`, and CI checks the gate in the production image (`.github/workflows/ci.yml`, "Check the learning-memory gate") |
| Observability | `app/core/metrics.py` (Prometheus), worker task metrics, `app/core/safe_log.py` (exceptions logged by type and place only) | Built |

**Routes served** (`app/main.py`, `app/api/v1/`): `/api/v1/health`, `/api/v1/system/{health,ready}`; `/v1/ingest/{dispatch,tasks/{id},scope-assignments}`; `/v1/ai/{chat,explain,generate-questions,retrieve,tree/generate,read/start,memory/query,memory/upsert}`.

## 3. Quality and evidence

- **Retrieval quality is not settled.** The PRD records 94.6% hybrid recall@5 on the controlled set and 62.5% (25 of 40) on real first-semester FUTA material, with 1,459 scanned pages still needing OCR (`build-backend` `docs/BUILD_MASTER_PRD.md` §0.6). This file didn't re-measure them.
- **CI gates** (`.github/workflows/backend.yml`): quality checks (pre-commit and explicit checks), a security scan (pip-audit, bandit), the full suite against PostgreSQL with pgvector and a migration review, real-storage fetch tests, and a retrieval quality gate that must not regress below the committed baseline (`data/retrieval_eval/v1`).
- Evaluation tooling and results: `docs/p3-5-tier2-evaluation.md`, `docs/p3-2-extraction-baseline.md`.

## 4. Gates

| Gate | State |
|---|---|
| Learning Memory | Closed. `GATE_OPEN = False` (`app/core/learning_memory_gate.py`) |
| External providers for student passages | No provider approved. `MODEL_PROVIDERS_APPROVED` is empty by default. G-33 isn't approved for any provider (the records are drafts in `build-backend`) |
| Retrieval of a document | Its scope must be acknowledged and it must not be denied (`app/core/rag/scope.py`) |

## 5. Known gaps (checked, not assumed)

1. **Row-level security:** none. No RLS policy or tenant `SET LOCAL` exists in `app/` or `alembic/`; tenant isolation is by scope filter and query predicates. (The only `SET LOCAL` calls tune HNSW search, in `app/services/rag/pg_index.py`.)
2. **Database pool:** set in `app/db/session.py` (`pool_size=10`, `max_overflow=20`, `pool_pre_ping=True`; no `pool_recycle`).
3. **Provider portfolio:** only `openai`, `deterministic` and `mock` exist; the Gemini adapter is open.
4. **AI-10, agents and tools** are built but unrouted.
5. **Offline AI** has no code on `main`. The offline benchmark runs in its own lane and isn't merged.
6. **Retrieval quality** on real material, and OCR coverage of scanned pages, remain the main quality work (section 3).
7. **Conversations** (M3-F's AI side, the tutor-turn route) are not on `main`.

## 6. What this file no longer says

An earlier version described a service with no vector store and no task queue, a JWT and exam layer, a WebSocket manager, `app/api/v1/exams.py`, placeholder retrieval, 208 passing tests, and a roadmap dated 2026-08 to 2026-12. None of that is true of `main`: the exam and JWT runtime were removed in M4, pgvector and the task worker are built, and the roadmap was overtaken. The history is in git.

## 7. Where the roadmap is

In `build-backend`:
- `docs/BUILD_IMPLEMENTATION_PLAN.md`: phases, releases and dependencies.
- `docs/build-master-execution-blueprint.md`: the capability map (AI-01 to AI-14) and the parallel lanes.
- `docs/milestone-register.md`: the tracker.
