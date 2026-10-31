"""Central exception handling."""

from fastapi.responses import JSONResponse

from app.core.logging import logger


async def learning_memory_disabled_handler(request, exc):
    """A closed learning-memory gate is an expected refusal, not a fault."""
    return JSONResponse(
        status_code=503,
        content={
            "success": False,
            "message": "Learning memory is unavailable",
            "data": {},
        },
    )


async def app_exception_handler(request, exc):
    """Catch-all exception handler for unhandled errors."""
    request_id = getattr(request.state, "request_id", "unknown")

    logger.error(f"Unhandled exception: {exc!s} [request_id={request_id}]")

    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "message": "Internal server error",
            "data": {},
        },
    )
