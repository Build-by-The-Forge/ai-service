"""Central exception handling."""

from fastapi.responses import JSONResponse

from app.core.logging import logger
from app.core.safe_log import describe


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
    """Catch-all exception handler for unhandled errors.

    It logs the exception's type and where it was raised, never its text: an
    exception's message can hold a student passage, a prompt, or a provider's
    response body (a provider SDK puts the body in its errors), and none of
    those is ours to log. No traceback is attached, since its last line is
    the message again."""
    request_id = getattr(request.state, "request_id", "unknown")

    logger.error("Unhandled exception: %s [request_id=%s]", describe(exc), request_id)

    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "message": "Internal server error",
            "data": {},
        },
    )
