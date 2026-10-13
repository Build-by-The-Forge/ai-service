# Phase 1 — Milestone 4: AI-service legacy inventory

**Purpose:** plan §6.1. Every runtime use of the old AI-service identity and exam domain, classified before anything is removed.
**Baseline:** `feat/m3-boundary` at `1ee9778`. After M3-10, none of these routes is mounted unless `LEGACY_USER_ROUTES_ENABLED=true`, which is development only.
**Classes:** REMOVE · REPLACE WITH BACKEND CONTEXT · REMODEL AS AI DOMAIN · TEST FIXTURE ONLY · DOCUMENTATION ONLY

## 1. Who calls the legacy surface

- **The backend:** nothing. `build-backend/app` has no call to `/auth/*`, `/exams`, `/api/v1/exams`, `/api/v1/ai/exam*`, fusion exam or `ws /ws/exam/*`. Its only AI paths are `/v1/ingest/*` and `/v1/ai/{chat,explain,generate-questions}`, both signed (M3).
- **Clients:** nothing in M3 or later. These routes are unmounted by default and cannot authenticate against the shared database: AI-service's `users` model expects `school_id` and `role` columns that the backend's `users` table does not have (audit A-4).

## 2. Identity (plan §6.2): AI-service must not authenticate users

| Code | What it is | Class |
|---|---|---|
| `app/auth/routes.py` (`/auth/login`, `/refresh`, `/me`, `/logout`) | AI-service's own user login against its legacy `users` model | REMOVE |
| `app/auth/token.py`, `app/auth/dependencies.py`, `app/auth/permissions.py`, `app/core/auth.py` | HS256 JWTs (`JWT_SECRET`), `school_id`/`role` claims, student/teacher/admin checks | REMOVE |
| `app/core/token_verifier.py` | A placeholder "verifier" that returns a fixed payload (`school_id: 1, role: student`) | REMOVE |
| `app/core/rbac.py` | Legacy role model (`student`/`teacher`/`admin`); used only by `app/auth` | REMOVE |
| Settings `JWT_SECRET`, `JWT_PRIVATE_KEY`, `JWT_PUBLIC_KEY`, `JWT_ALGORITHM`, `JWT_AUDIENCE`, `LEGACY_USER_ROUTES_ENABLED` | Exist only for the above | REMOVE |
| `current_school_id` / `set_school_id` / `get_school_id` in `app/core/tenant.py` | Integer tenant for the legacy routes; M3 paths already use the string scope | REMOVE (A-5) |

After M4, AI-service authenticates **services** only (BUILD Service Signature v1). Users are authenticated, and their tenant and membership resolved, by the backend.

## 3. Exam runtime (plan §6.3): the backend owns the CBT lifecycle

| Code | What it is | Class |
|---|---|---|
| `app/api/v1/exams.py` (`/exams`, `/api/v1/exams`) | Exam CRUD against AI-service's `exams` table | REMOVE (backend owns assessments) |
| `app/api/v1/ai/exam.py`, `exam_submit.py` | Exam start and submission/scoring | REMOVE (backend owns attempts, answers, scores) |
| `app/api/v1/ws/exam_ws.py` | Exam websocket: session state, timer, anti-cheat | REMOVE |
| `app/services/exam/{exam_engine, session_service, session_state, question_service, scoring, timer, redis_timer, redis_session_store, anti_cheat}.py` | Exam sessions, timers, answers, scoring, proctoring | REMOVE |
| `app/services/realtime/{pubsub, sync_bridge}.py` | Real-time exam session sync; used only by the exam websocket | REMOVE |
| `app/schemas/exam.py` | Exam request/response schemas | REMOVE |
| `app/api/v1/ai/fusion_exam.py` | Route returning a personalised question set for a user id taken from the request | REMOVE (route); a client-chosen user id is not acceptable (A-1) |
| `app/services/exam/adaptive_engine.py` | Difficulty for a topic from the learner's mastery memory | **REMODEL AS AI DOMAIN**: difficulty analysis is AI intelligence (§6.3). Moves to `app/services/generation/`. Memory is already scoped to the verified tenant and user. |
| `app/services/exam/fusion_engine.py` | Adaptive difficulty + question generation | **REMODEL AS AI DOMAIN**: personalised question generation. Moves to `app/services/generation/`; M5 decides its contract. |

## 4. Legacy ORM models

| Model (table) | Class | Note |
|---|---|---|
| `User` (`users`), `School` (`schools`) | REMOVE | The backend owns `users` and `tenants`. AI-service never had a migration for them in the shared database (excluded by `app/db/ownership.py`). |
| `Exam` (`exams`), `ExamSession` (`exam_sessions`), `AnswerSheet` (`answer_sheets`) | REMOVE | The backend owns assessments, attempts and answers. |
| `Question` (`questions`, `question_bank.py`) | REMOVE | The backend owns `questions`. Since M3-13, `build_ai_app` cannot even read the table. |

**No database change.** AI-service's Alembic chain only ever owned `learning_memory` and `session_memory` (`ai_service_alembic_version`). Removing these models changes no table and needs no migration. `app/db/session.initialize_database` (test-only `create_all`) then creates only AI-owned tables.

## 5. Kept, with reasons

| Code | Class | Reason |
|---|---|---|
| `app/core/tools/validator.py`: `"school_id"` in the tool tenant-boundary check | REPLACE WITH BACKEND CONTEXT (already done) | It compares any `tenant_id`/`school_id` tool argument with the authoritative context tenant and refuses a mismatch. The legacy key stays as a defensive alias, so a tool cannot be steered by it. |
| `app/core/dependencies.py` (`get_db`) | Kept | A generic session dependency, not legacy-specific. |
| `app/services/ai/tenanted_retriever.py` comment mentioning `school_id` | DOCUMENTATION ONLY | Comment corrected. |
| `docs/*` audit mentions | DOCUMENTATION ONLY | Historical records; not edited. |

## 6. Tests

| Tests | Class |
|---|---|
| `tests/integration/test_auth_exam.py`, `tests/unit/test_rbac_and_jwt.py`, `test_core_token_verifier.py`, `test_exam_engine.py`, `test_exam_helpers.py`, `test_legacy_user_routes.py` | REMOVE with the code they test |
| Legacy cases inside `test_additional_coverage.py`, `test_api_and_auth_coverage.py`, `test_extra_coverage.py`, `test_tenant.py` | REMOVE those cases only; the others stay |
| `test_adaptive_memory.py`, `test_fusion_engine.py` | Kept, retargeted at the new module paths |
| `tests/integration/conftest.py` `seed_user` | TEST FIXTURE ONLY → REMOVE with its only caller |

## 7. Order of removal

1. Remodel: move the adaptive and fusion engines to `app/services/generation/`.
2. Exam runtime: routes, websocket, services, realtime bridge, schemas.
3. User authentication: `/auth`, JWT, RBAC, token verifier, the legacy flag and settings.
4. Legacy models and the integer `school_id` context.
5. Compose and configuration references in build-backend.
