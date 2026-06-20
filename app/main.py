from fastapi import FastAPI
from app.core.config import settings
from app.core.logging import setup_logging
from app.core.metrics import setup_metrics
from app.core.errors import app_exception_handler
from app.core.middleware.tenant import TenantMiddleware
from app.core.middleware.request_id import RequestIDMiddleware
from app.api.v1.router import api_router
from app.db.session import engine
from app.db.base import Base


def create_app() -> FastAPI:
    setup_logging()

    app = FastAPI(
        title="AI Service - Build Core",
        version="1.0.0",
        description="AI engine for CBT + Learning Platform",
    )

    # Middleware stack (order matters)
    app.add_middleware(RequestIDMiddleware)
    app.add_middleware(TenantMiddleware)

    # Setup metrics
    setup_metrics(app)

    # Exception handlers
    app.add_exception_handler(Exception, app_exception_handler)

    # Routes
    app.include_router(api_router, prefix="/api/v1")

    @app.on_event("startup")
    async def startup():
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    return app


app = create_app()


@app.get("/")
def root():
    return {
        "success": True,
        "data": {
            "service": "ai-service",
            "status": "running"
        },
        "message": "AI Service is live"
    }
