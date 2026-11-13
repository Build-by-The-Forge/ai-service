"""Central exception handling."""

from fastapi.responses import JSONResponse

from app.core.logging import logger
from app.core.models.errors import ModelGatewayError
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


async def model_gateway_error_handler(request, exc):
    """A model execution failure is an outage, not the caller's fault (AI-01
    decision P3-1): 503 AI_SERVICE_UNAVAILABLE with a fixed body, which the
    backend already treats as the AI service being unavailable. It covers a
    provider down or timed out, a rate limit, an open circuit, no approved
    provider (an empty MODEL_PROVIDERS_APPROVED fails closed), exhausted
    fallbacks, an exhausted budget and the request deadline. Invalid input
    never gets here: the route refuses it with 422 before any model call.
    The body carries no provider text, message, prompt or passage; the log
    line carries safe fields only."""
    request_id = getattr(request.state, "request_id", "unknown")
    status_code = exc.status_code if isinstance(exc, ModelGatewayError) else None
    provider = exc.provider if isinstance(exc, ModelGatewayError) else None
    logger.warning(
        "Model execution failed, answered 503: %s provider=%s status=%s [request_id=%s]",
        type(exc).__name__,
        provider,
        status_code,
        request_id,
    )
    return JSONResponse(
        status_code=503,
        content={
            "success": False,
            "code": "AI_SERVICE_UNAVAILABLE",
            "message": "AI service unavailable",
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
