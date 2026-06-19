from fastapi import FastAPI
from app.core.config import settings
from app.core.logging import setup_logging
from app.api.v1.router import api_router

def create_app() -> FastAPI:
    setup_logging()

    app = FastAPI(
        title="AI Service - Build Core",
        version="1.0.0",
        description="AI engine for CBT + Learning Platform",
    )

    app.include_router(api_router, prefix="/api/v1")

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
