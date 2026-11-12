from fastapi import APIRouter

from app.core.logging import logger
from app.core.redis import get_redis_client
from app.core.safe_log import describe
from app.db.session import get_engine

router = APIRouter()


@router.get("/health")
async def health():
    return {
        "success": True,
        "data": {"status": "ok"},
        "message": "healthy",
    }


@router.get("/ready")
async def readiness():
    """Return readiness status with dependency checks."""
    checks = {
        "database": False,
        "redis": False,
    }

    try:
        eng = get_engine()
        async with eng.connect() as conn:
            await conn.run_sync(lambda c: None)
        checks["database"] = True
    except Exception as exc:
        # Type and place only: a connection error can name hosts, users and settings.
        logger.error("Database readiness check failed: %s", describe(exc))

    try:
        await get_redis_client().ping()
        checks["redis"] = True
    except Exception as exc:
        logger.error("Redis readiness check failed: %s", describe(exc))

    all_ready = all(checks.values())

    return {
        "success": all_ready,
        "data": checks,
        "message": "ready" if all_ready else "not ready",
    }
