# AI Service - Build Core

Production-grade AI service for CBT + Learning Platform.

## Architecture

```
Internet
   ↓
Nginx (Reverse Proxy)
   ↓
FastAPI (ai-service)
   ↓
PostgreSQL + Redis
   ↓
AI/RAG/OCR/Tree/CBT
```

## Features

### Platform Core
- **FastAPI** - Modern async web framework with OpenAPI/Swagger
- **Multi-tenancy** - School-scoped request context + database isolation
- **RBAC** - Role-based access control (student/teacher/admin)
- **JWT** - Token verification ready for external auth services

### Observability
- **Structured Logging** - JSON log format for production parsing
- **Request Tracing** - Unique request ID on every request
- **Metrics** - Prometheus `/metrics` endpoint
- **Health Checks** - `/api/v1/system/health` and `/api/v1/system/ready` with dependency probes

### AI/ML Pipelines
- **RAG** - Retrieval-Augmented Generation with tenant-aware retrieval
- **OCR** - Document parsing and text extraction foundation
- **Tree Generation** - Knowledge tree generation and traversal
- **AI Orchestration** - LLM integration with memory + retrieval context
- **CBT Engine** - Computer-based testing with adaptive difficulty
- **Exam Engine** - Real-time exam sessions with Redis pub/sub

### Reliability
- **Rate Limiting** - Configurable per-endpoint via slowapi
- **Error Handling** - Central exception handler with request context
- **Database Pooling** - Async SQLAlchemy with connection pooling
- **Redis Resilience** - Distributed session store and pub/sub
- **Readiness Probes** - Actual DB + Redis connectivity checks

### Deployment
- **Docker** - Multi-stage Dockerfile with entrypoint migration
- **Compose** - PostgreSQL + Redis + Nginx orchestration
- **Makefile** - Simple targets: `make up`, `make down`, `make logs`
- **Kubernetes-Ready** - Health checks and structured configuration
- **CI/CD** - GitHub Actions with linting, type checking, tests, coverage

## Setup

### Local Development

```bash
# Clone and enter
git clone <repo>
cd ai-service

# Create environment
cp .env.example .env

# Install dependencies
pip install -r requirements.txt

# Run with Uvicorn
uvicorn app.main:app --reload
```

### Docker Compose

```bash
# Start full stack
make up

# View logs
make logs

# Stop services
make down

# Run migrations
make migrate
```

## API Endpoints

### System
- `GET /` - Root status
- `GET /api/v1/system/health` - Liveness probe
- `GET /api/v1/system/ready` - Readiness probe with dependency checks
- `GET /metrics` - Prometheus metrics

### Health & Legacy
- `GET /api/v1/health/` - Legacy health check

### AI Chat
- `POST /api/v1/ai/chat` - Conversational AI with context
- `GET /api/v1/ai/explain` - Subject explanation
- `GET /api/v1/ai/generate-questions` - Question generation

### RAG & Retrieval
- `GET /api/v1/ai/retrieve/` - Tenant-scoped document retrieval

### Tree & Knowledge
- `POST /api/v1/ai/tree/generate` - Generate learning tree
- `GET /api/v1/ai/read/start` - Reading mode

### Exams
- `POST /api/v1/ai/exam/start` - Start exam session (requires student role)
- `POST /api/v1/ai/exam/submit` - Submit answers (requires student role)
- `POST /api/v1/ai/generate` - Generate AI exam (requires teacher/admin role)

### WebSocket
- `WS /ws/exam/{session_id}` - Real-time exam state sync with tenant scoping

## Environment Variables

### Development (`.env`)
```
APP_NAME=build-ai-service
ENV=development
DATABASE_URL=postgresql+asyncpg://postgres:postgres@postgres/build_ai
REDIS_URL=redis://redis:6379
OPENAI_API_KEY=<your-key>
LOG_LEVEL=INFO
```

### Production (`.env.prod`)
```
ENV=production
LOG_LEVEL=WARNING
```

### Optional
```
JWT_PUBLIC_KEY=<your-rsa-public-key>
JWT_ALGORITHM=RS256
JWT_AUDIENCE=ai-service
```

## Project Structure

```
app/
├── api/
│   └── v1/
│       ├── ai/              # AI/LLM endpoints
│       │   ├── chat.py
│       │   ├── exam.py
│       │   ├── fusion_exam.py
│       │   ├── generate_questions.py
│       │   ├── retrieve.py
│       │   ├── tree.py
│       │   └── ...
│       ├── ws/              # WebSocket
│       │   └── exam_ws.py
│       ├── system.py        # Health/ready endpoints
│       ├── health.py
│       └── router.py        # Route aggregation
├── core/
│   ├── auth.py              # RBAC guards (header-based)
│   ├── config.py            # Settings + JWT config
│   ├── errors.py            # Central error handler
│       ├── logging.py        # JSON structured logging
│   ├── metrics.py           # Prometheus setup
│   ├── rate_limit.py        # slowapi limiter
│   ├── redis.py             # Redis client
│   ├── rbac.py              # Role definitions
│   ├── tenant.py            # Multi-tenant context
│   ├── token_verifier.py    # JWT verification
│   └── middleware/
│       ├── request_id.py    # Request tracing
│       └── tenant.py        # Tenant header extraction
├── db/
│   ├── base.py              # SQLAlchemy declarative base
│   ├── models/
│   │   ├── exam.py
│   │   ├── question.py
│   │   ├── user.py          # With school_id FK
│   │   ├── school.py
│   │   ├── learning_memory.py
│   │   └── ...
│   └── session.py           # Async session factory
├── schemas/
│   ├── common.py            # APIResponse base
│   └── ...
├── services/
│   ├── ai/
│   │   ├── dependencies.py  # AI orchestrator DI
│   │   ├── llm.py           # LLMProvider interface
│   │   ├── openai.py        # OpenAI implementation
│   │   ├── orchestrator.py  # Chat/explain/generation
│   │   └── tenanted_retriever.py
│   ├── exam/
│   │   ├── exam_engine.py
│   │   ├── session_service.py
│   │   ├── redis_session_store.py
│   │   ├── redis_timer.py
│   │   └── ...
│   ├── memory/              # Student learning memory
│   │   └── memory_service.py
│   ├── rag/                 # Retrieval-Augmented Generation
│   │   ├── retrieval.py
│   │   └── vector_store.py
│   ├── tree/                # Knowledge tree
│   ├── realtime/            # Real-time sync
│   │   ├── pubsub.py
│   │   └── sync_bridge.py
│   └── generation/          # Question/exam generation
└── main.py                  # FastAPI app factory

docker/
├── entrypoint.sh            # Migration + startup
└── nginx/
    └── nginx.conf           # Reverse proxy config

tests/                       # Test suite (pytest)

.github/
└── workflows/
    └── backend.yml          # CI/CD pipeline

Dockerfile
docker-compose.yml
Makefile
requirements.txt
.env
.env.prod
README.md
```

## Development

### Tests
```bash
pytest
pytest --cov=app tests/
```

### Linting
```bash
ruff check app/
black app/ --check
```

### Type Checking
```bash
mypy app/
```

### Code Formatting
```bash
black app/
```

## Deployment

### Local Docker
```bash
make up
```
Visit `http://localhost` for Nginx → FastAPI.

### Health Checks
- **Liveness**: `curl http://localhost/api/v1/system/health`
- **Readiness**: `curl http://localhost/api/v1/system/ready`

### Logs
```bash
make logs
```

### Migrations
```bash
make migrate
# Or manually in container:
docker compose exec ai-service alembic upgrade head
```

### Database
```bash
# Access PostgreSQL
docker compose exec postgres psql -U postgres -d build_ai

# Access Redis
docker compose exec redis redis-cli
```

## Production Status

### ✅ Implemented
- **Architecture**: FastAPI + SQLAlchemy + Redis + PostgreSQL
- **Database**: Async ORM with models for all entities
- **Multi-tenancy**: Request-scoped tenant context + school_id enforcement
- **Authentication**: RBAC guards + JWT verification ready
- **Observability**: Structured JSON logging + request IDs + metrics + health checks
- **RAG**: Tenant-aware retrieval pipeline
- **OCR**: Document parsing foundation
- **Tree**: Knowledge graph generation
- **AI Orchestration**: LLM + memory + retrieval coordination
- **CBT Engine**: Adaptive question selection + scoring
- **Real-time**: WebSocket exam sync with Redis pub/sub
- **Deployment**: Docker + Compose + Nginx + Makefile
- **CI/CD**: GitHub Actions linting + tests + build
- **Error Handling**: Central exception handler
- **Rate Limiting**: Per-endpoint configurable limits

### 🎯 Next Engineering Steps
1. **Implementation Verification** - End-to-end integration tests
2. **Repository Cleanup** - ADRs (Architecture Decision Records) for key patterns
3. **Load Testing** - Benchmark horizontal scaling behavior
4. **Security Audit** - Penetration testing + OWASP validation
5. **Documentation** - API contract docs + deployment runbooks

### 🚀 Future Microservices
```
ai-service (core + exam)
ocr-worker (document processing)
embedding-worker (vector generation)
retrieval-worker (RAG optimization)
```

## Security Notes

- Secrets (JWT, API keys, DB passwords) must be in environment variables
- HTTPS required for production (configure in Nginx)
- CORS configured by default (adjust as needed)
- Rate limiting enabled on AI endpoints
- Tenant isolation enforced at request + database layer

## Support

For issues, questions, or contributions:
1. Review `Architecture Decision Records` in `/docs/adr`
2. Check test suite in `/tests`
3. Review health/readiness responses
4. Check structured logs for context

---

**Status**: Production-ready platform with AI, RAG, OCR, CBT, real-time sync, multi-tenancy, and observability.
