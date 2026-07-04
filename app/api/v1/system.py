from fastapi import APIRouter
from app.db.session import get_engine
from app.core.redis import redis_client
from app.core.logging import logger

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
        logger.error(f"Database readiness check failed: {exc}")

    try:
        await redis_client.ping()
        checks["redis"] = True
    except Exception as exc:
        logger.error(f"Redis readiness check failed: {exc}")

    all_ready = all(checks.values())

    return {
        "success": all_ready,
        "data": checks,
        "message": "ready" if all_ready else "not ready",
    }
