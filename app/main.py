from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.v1.ai.router import ai_router
from app.api.v1.ingest import router as ingest_router
from app.api.v1.router import api_router
from app.core.config import STRICT_ENVIRONMENTS, settings
from app.core.errors import app_exception_handler
from app.core.logging import setup_logging
from app.core.metrics import setup_metrics
from app.core.middleware.request_id import RequestIDMiddleware
from app.core.middleware.service_auth import ReplayGuard, SignedServiceRequestMiddleware
from app.core.replay_guard import RedisReplayGuard
from app.core.service_signature_v1 import AI_SERVICE_ID, BACKEND_SERVICE_ID, ServiceVerifier
from app.db.session import reset_engine

# Routes reachable only with a verified BUILD Service Signature v1 request.
SIGNED_PATH_PREFIXES = ("/v1/ingest/", "/v1/ai/")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup never creates schema: the database is shared with the Build
    # backend, and AI-owned tables come only from Alembic (docker/entrypoint.sh
    # runs `alembic upgrade head` before starting the app).
    setup_logging()

    yield

    await reset_engine()


def _inbound_verifier() -> ServiceVerifier | None:
    secrets = settings.backend_to_ai_verification_secrets
    if not secrets:
        return None  # development without secrets: every signed route refuses
    return ServiceVerifier(accepted_sender=BACKEND_SERVICE_ID, secrets=secrets)


def create_app(replay_guard: ReplayGuard | None = None) -> FastAPI:
    # Interactive docs and the OpenAPI document are for development and
    # testing; staging and production expose neither (M6).
    docs = settings.APP_ENVIRONMENT not in STRICT_ENVIRONMENTS
    app = FastAPI(
        title="AI Service - Build Core",
        version="1.0.0",
        description="AI engine for CBT + Learning Platform",
        lifespan=lifespan,
        docs_url="/docs" if docs else None,
        redoc_url="/redoc" if docs else None,
        openapi_url="/openapi.json" if docs else None,
    )

    # Middleware stack (order matters)
    app.add_middleware(RequestIDMiddleware)
    # Added last, so it runs first: nothing else sees an unverified request.
    app.add_middleware(
        SignedServiceRequestMiddleware,
        receiver_id=AI_SERVICE_ID,
        verifier=_inbound_verifier(),
        replay_guard=replay_guard
        or RedisReplayGuard.connect(
            settings.REDIS_URL, timeout=settings.SERVICE_REPLAY_TIMEOUT_SECONDS
        ),
        applies_to=lambda path: path.startswith(SIGNED_PATH_PREFIXES),
    )

    # Setup metrics
    setup_metrics(app)

    # Exception handlers
    app.add_exception_handler(Exception, app_exception_handler)

    # Routes
    app.include_router(api_router, prefix="/api/v1")
    app.include_router(ingest_router, prefix="/v1/ingest")
    app.include_router(ai_router, prefix="/v1/ai")

    return app


app = create_app()


@app.get("/")
def root():
    return {
        "success": True,
        "data": {"service": "ai-service", "status": "running"},
        "message": "AI Service is live",
    }
