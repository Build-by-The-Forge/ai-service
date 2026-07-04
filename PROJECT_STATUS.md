# Project Status

## 🟢 Green
These areas are in a good state and provide a solid foundation for continued development.

- ✅ Docker Compose starts PostgreSQL and Redis
- ✅ Seed script works
- ✅ Alembic migrations work
- ✅ Integration tests pass
- ✅ Health endpoints pass
- ✅ Authentication works
- ✅ Redis connectivity works
- ✅ RAG pipeline works
- ✅ Database engine lifecycle cleanup is in place
- ✅ .dockerignore is configured
- ✅ Black formatting is standardized
- ✅ CI pipeline exists

## 🟡 Yellow
These should be addressed before merging into main.

### 1. Test coverage
- Current coverage: 53.86%
- Target coverage: 85%
- The main gap is that the repo currently relies mostly on integration tests.
- The next step is to add focused unit tests for services, auth, RAG, OCR, generation, and utilities.

### 2. MyPy
- Current status: 11 errors
- These are manageable but should be cleaned up.
- Categories include:
  - missing stubs for jose, passlib, and fitz
  - real typing issues such as optional limiter handling, server_onupdate, RBAC, and tenant-aware retrieval service
  - assignment mismatches that may indicate annotation bugs

### 3. Ruff and Black
- Ruff issues are mostly resolved.
- Black is in good shape and should be used as a pre-commit/CI check rather than a one-off local formatting exercise.

### 4. Docker context hygiene
- The new .dockerignore helps prevent large virtualenv artifacts from being sent into the Docker build context.
- Keep this maintained as the project grows.

### 5. SQLAlchemy shutdown noise on Windows
- The asyncpg event-loop shutdown warnings are common on Windows under pytest.
- They are not currently breaking the suite, so they are not the top priority unless they become functional failures.

## 🔴 Red
These are the highest-risk items if the repository is to be treated as production-grade.

- Coverage is still too low for a mature backend service.
- MyPy is not yet fully clean.
- The test suite is still weighted toward integration coverage rather than unit coverage.
- CI should continue to enforce quality gates consistently for ruff, black, mypy, pytest, coverage, and docker build.

## Recommended next priorities
1. Raise coverage above 85%
2. Fix the remaining MyPy issues completely
3. Add unit tests around core services and auth/RAG utilities
4. Keep CI enforcement strict for quality and security checks
5. Add pre-commit hooks for ruff and black

## Stabilization roadmap

### Phase 1 — Finish engineering quality
- Fix all remaining MyPy errors until `mypy app` passes with no issues.
- Raise coverage from 53.86% to at least 85% by adding unit tests for AI orchestrator, exam engine, fusion engine, retrieval service, OCR service, JWT handling, RBAC, and utilities.
- Make CI green end-to-end so every PR runs ruff, black, mypy, alembic, tests, coverage, and docker build.

### Phase 2 — Improve developer experience
- Add pre-commit hooks to run ruff fixes, black, and mypy before commits.
- Add a Makefile or justfile with commands for dev, test, lint, ci, seed, and migrate workflows.
- Document the authentication flow, AI orchestrator, OCR pipeline, RAG pipeline, exam flow, WebSocket flow, tenant isolation, and database design.

### Phase 3 — Prepare for scale
- Add structured logging with request ID, tenant ID, user ID, latency, and status code.
- Add metrics for AI latency, OCR latency, Redis hits, database queries, exam generation time, and RAG retrieval time.
- Expand health checks to include OpenAI, Redis, Postgres, vector store, disk space, and memory.
- Introduce basic error monitoring such as Sentry.

### Phase 4 — Security audit
- Review JWT expiry and refresh-token handling.
- Validate password hashing and RBAC enforcement.
- Review rate limiting, prompt injection protection, file upload validation, SQL injection resistance, and tenant isolation.

### Phase 5 — Performance
- Benchmark exam generation, question generation, OCR, RAG, login, and exam submission.
- Track P50, P95, and P99 latency for the most important paths.

### Phase 6 — Codebase cleanup
- Keep service modules focused and avoid growing them into monolithic files.
- Continue reorganizing modules under clear service boundaries such as ai, rag, ocr, generation, exam, auth, memory, and tree.

### Phase 7 — Release v1.0
- Tag a stable release such as `v1.0.0`.
- Publish a changelog and freeze API contracts that the frontend or future repositories will depend on.
- Review and publish OpenAPI documentation.

## Overall assessment
The repository is in a strong position for a first backend project of this scope. The core functionality is working, the infrastructure is in place, and the development foundation is solid. The remaining work is mainly about engineering quality, maintainability, and test maturity.
