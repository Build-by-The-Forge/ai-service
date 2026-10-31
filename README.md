# AI Service - Build Core

The intelligence subsystem of BUILD: retrieval, explanation, question generation, learning memory and document ingestion. It runs behind the backend.

## Architecture

```
Client
   ↓  (user authentication, tenant, membership, permission)
build-backend
   ↓  signed request (BUILD Service Signature v1)
ai-service ──→ PostgreSQL (AI-owned tables only) + Redis
   ↑
   └── signed ingestion callbacks to the backend
```

AI-service **does not authenticate users** and has no client-facing routes. It accepts only requests signed by the backend:
- HMAC-SHA256 over a canonical request;
- directional secrets;
- ±60 s timestamp window;
- Redis replay protection.

The signed context carries the tenant, the user and the one permission the backend checked. Tenant and user are never taken from a request body, header or query. The backend owns users, tenants, memberships, questions, assessments, attempts and results. See `build-backend/docs/phase1-m3-boundary-design.md`.

## Features

### Boundary
- **Signed service requests only**: `/v1/ai/*` and `/v1/ingest/*` refuse anything else with 401. Redis down → 503 (fail closed). Bodies are capped at 1 MiB.
- **Context narrows only**: each route declares one permission; a context without it gets 403.
- **Request scope**: tenant and user from the verified context. Memory and retrieval fail closed without it.

### Intelligence
- **Chat**: LLM answer with tenant-scoped retrieval and the signed user's learning memory.
- **Explain** and **question generation** (1–50 questions).
- **Retrieval**: tenant-scoped hybrid retrieval (placeholder results until the real index exists).
- **Knowledge tree** generation and **reading path**.
- **Adaptive difficulty** and **personalised generation** (`app/services/generation/`).
- **Ingestion worker**: consumes dispatched tasks and reports signed callbacks to `BACKEND_CALLBACK_URL`. Phase 1 has no real pipeline, so tasks fail with `INGESTION_PIPELINE_UNAVAILABLE`.
- **Model provider**: `MODEL_PROVIDER=openai` (default) or `deterministic`, an offline stand-in for development and tests that is refused in staging/production.

### Observability
- JSON logging, request IDs, Prometheus `/metrics`.
- The separate task worker exposes its task-transition, callback-delivery, and stream-reclaim metrics at `/metrics` on internal port `9102`. Worker transition logs include job/task/run/attempt identity and exclude exception messages and credentials.
- `/api/v1/system/health` and `/api/v1/system/ready` (unauthenticated; reveal no configuration).

## Setup

```bash
cp .env.example .env            # or set variables in the environment
pip install -r requirements-dev.txt
alembic upgrade head            # AI-owned tables only: learning_memory, session_memory
uvicorn app.main:app --reload   # API
python -m app.worker            # task worker (separate process)
```

To run both services together, use the combined development topology in build-backend: `deploy/compose.m3.yml`, run with `scripts/m3_boundary_gate.sh`. It has least-privilege database roles, an unpublished AI-service and the worker.

```bash
make lint        # ruff, black --check, mypy
make test
make precommit   # install the git hooks (ruff --fix, black, mypy)
```

## API

All routes below except System require a signed backend request.

### System (unauthenticated)
- `GET /`, `GET /api/v1/system/health`, `GET /api/v1/system/ready`, `GET /api/v1/health/`, `GET /metrics`

### AI (signed, user context)
| Route | Permission | Body |
|---|---|---|
| `POST /v1/ai/chat` | `ai.use` | `{message}` (≤ 4000) |
| `POST /v1/ai/explain` | `ai.use` | `{subject}` (≤ 4000) |
| `POST /v1/ai/generate-questions` | `ai.questions.generate` | `{topic, count}` (count 1–50) |
| `POST /v1/ai/retrieve` | `ai.use` | `{query}` |
| `POST /v1/ai/tree/generate` | `ai.use` | `{course_title, topics}` |
| `POST /v1/ai/read/start` | `ai.use` | `{root}` |

Bodies reject unknown fields. Naming `tenant_id`, `user_id`, `membership_id` or `role` is 422.

### Ingestion (signed)
- `POST /v1/ingest/dispatch`: user context with `content.job.manage`.
- `GET /v1/ingest/tasks/{external_task_id}`: system context; answers only for the context's tenant.

## Configuration

| Setting | Rule |
|---|---|
| `APP_ENVIRONMENT` | `development` / `testing` / `staging` / `production`; strict validation in the last two |
| `BACKEND_TO_AI_SECRET`, `AI_TO_BACKEND_SECRET` (+ `_PREVIOUS`) | Required in staging/production; ≥ 32 bytes; the two directions must differ; no known development values |
| `BACKEND_CALLBACK_URL` | Fixed `…/content/ingestion/callbacks`; https in staging/production |
| `REDIS_URL` | Replay protection, task queue and task records |
| `DATABASE_URL` | Runtime role with DML on the AI-owned tables only |
| `MODEL_PROVIDER`, `OPENAI_API_KEY` | See above |
| `LEARNING_MEMORY_ENABLED` | Off by default. Refused at startup while the learning-memory production gate is closed, except in development and testing outside the production image. See `app/core/learning_memory_gate.py` |
| `SERVICE_REPLAY_TIMEOUT_SECONDS` | Replay-guard Redis timeout, 0.05–5.0 (default 1.0) |

## Project structure

```
app/
├── api/v1/
│   ├── ai/            # signed /v1/ai routes (router.py mounts them)
│   ├── ingest.py      # signed dispatch + task status
│   ├── system.py, health.py
│   └── router.py      # unauthenticated system routes
├── core/
│   ├── service_signature_v1.py, service_context.py, service_call.py
│   ├── replay_guard.py, middleware/service_auth.py
│   ├── tenant.py      # verified request scope (tenant, user)
│   ├── callbacks/     # signed callbacks to the backend
│   ├── config.py
│   └── ...
├── db/                # learning_memory, session_memory (AI-owned)
├── services/
│   ├── ai/            # orchestrator, model providers
│   ├── generation/    # question generation, adaptive difficulty
│   ├── memory/, rag/, tree/
│   └── tasks/         # worker, task records, ingestion outcomes
├── main.py            # app factory
└── worker.py          # python -m app.worker
```

## Security notes

- Secrets come from the environment only. Production refuses missing, short, equal or default secrets.
- The service must not be reachable from clients. In the combined topology it is unpublished, and the backend is its only caller.
- Responses are not signed. Transport security (TLS/mTLS) is a Phase 1 M6 item.
