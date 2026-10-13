# Phase 1 — Milestone 4: Completion Report

**Milestone:** M4 — AI-Service Legacy Removal & Domain Boundary
**Status:** Complete (2026-10-10)
**Governing documents:** `docs/PHASE_1_PLAN.md` §6 (Workstream 4), §22 M4; `docs/phase1-m4-legacy-inventory.md` (classification, §6.1)

## 1. Verdict and evidence

AI-service no longer has a user-identity or exam domain. It authenticates services only, and its database model holds only AI-owned tables. Nothing was removed without first being classified, and nothing classified as AI intelligence was lost.

| Gate | Result |
|---|---|
| Real two-service gate: `build-backend/scripts/m3_boundary_gate.sh`, fresh PostgreSQL 17 + Redis 8, images rebuilt from the M4 code | **141 passed**, then C11 with Redis stopped: **3 passed**; exit 0. Includes the new M4 checks: every former identity or exam route is 404 on the running AI-service, and its OpenAPI lists only signed routes and system probes. |
| ai-service full suite, fresh PostgreSQL 17 + Redis 8 | 749 passed, 3 skipped, **2 failed — both pre-existing** (§5). The legacy `test_auth_exam` error is gone with the code it tested. |
| ai-service pre-commit (ruff, black, mypy) | Passed on every commit. |

## 2. What was removed (ai-service)

| Area | Commit |
|---|---|
| Inventory and classification (§6.1) | `1565431` |
| Adaptive difficulty and personalised generation moved to `app/services/generation/` (kept as AI domain) | `147ecae` |
| Exam runtime: `/exams`, `/api/v1/exams`, exam start/submit, fusion-exam route, exam websocket, `services/exam`, `services/realtime`, exam schemas | `0b6a083` |
| User authentication: `/auth/*`, HS256 JWT, role guards, legacy RBAC, the placeholder `TokenVerifier`; the `JWT_*` and `LEGACY_USER_ROUTES_ENABLED` settings | `96c7e4b` |
| Legacy ORM models `User`, `School`, `Exam`, `ExamSession`, `AnswerSheet`, `Question`; `scripts/seed.py`, which ran `create_all` and inserted legacy rows into any database; the integer `school_id` context (A-5) | `1b8ac90` |
| `python-jose`, `passlib`, `bcrypt` and their stubs | `04d047e` |
| README and PROJECT_STATUS rewritten to match the service | `6d56cae` |

Backend side:
- `LEGACY_USER_ROUTES_ENABLED` dropped from the combined topology (`aca189d`).
- Unused AI client stubs removed (`cbd3bfb`).
- Real-stack M4 check added (boundary suite).

About 1,900 lines of legacy code and tests were deleted. **No database change**: AI-service's Alembic chain never owned the legacy tables, and the ownership filter is unchanged.

## 3. Domain boundary after M4

| Owner | Data and responsibilities |
|---|---|
| **build-backend** | users, sessions, tenants, memberships, permissions; courses and curriculum; questions, assessments, attempts, answers, results; content sources, assets, ingestion jobs and their event ledger |
| **ai-service** (tables) | `learning_memory`, `session_memory`; its own `ai_service_alembic_version` |
| **ai-service** (non-relational, today) | task records, idempotency and the task queue (Redis); document intake, retrieval and vector-store contracts in code (`app/core/documents`, `app/core/rag`), with no persistent index yet (plan §13); agent and model-gateway runtime state |

This boundary is enforced, not just documented:
- **Database:** since M3-13, AI-service's runtime role has no privilege on any backend table (C24).
- **Code:** `Base.metadata` holds only the two AI-owned tables.
- **HTTP:** every AI-service route except the system probes requires a backend-signed request.

## 4. Kept on purpose

- `AdaptiveDifficultyEngine` and `CBTFusionEngine` are intelligence (plan §6.3), now in `app/services/generation/`. They have no route; M5 defines their contract.
- `app/core/tools/validator.py` still treats a `school_id` tool argument as a tenant claim and refuses it unless it equals the context tenant. This is a defensive alias, not a source of tenancy.
- The pre-commit mypy hook still lists the two removed type-stub packages. Changing that list rebuilds the hook environment, which Windows App Control blocks on the development machine. The stubs are inert.

## 5. Known issues (pre-existing; outside M4)

| Issue | Finding |
|---|---|
| `test_service_stubs.py::test_image_ocr_providers_return_data` | Fails on the pre-M3 baseline; unchanged. |
| `test_idempotency_store.py::test_redis_idempotency_store_operation_and_duplicates` | **Root cause confirmed:** `app/core/redis.py` holds one module-global asyncio Redis client. In a second event loop its calls raise "Event loop is closed", and `RedisIdempotencyStore` catches the error and **falls back to a per-process in-memory store**, so a duplicate goes undetected. In tests this is cross-loop pollution. In production it means **a Redis error silently degrades idempotency to per-replica memory**. Recorded for M6 (production hardening): idempotency must fail closed or be scoped per loop. This is very likely the unidentified Redis-backed failure seen in the M3-09 gate. |

## 6. Handover to M5

- AI contracts to formalise:
  - `/v1/ai/*` requests and responses;
  - memory writes (`learning_memory` is read, never written);
  - adaptive and personalised generation;
  - what retrieval, tree and read expose through the backend.
- Consolidate the two model stacks: `services/ai` `LLMProvider` and the `core/models` gateway.
