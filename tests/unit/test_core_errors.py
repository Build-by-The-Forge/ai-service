import logging

from fastapi.responses import JSONResponse

from app.core.errors import app_exception_handler


class DummyState:
    request_id = "test-request-id"


class DummyRequest:
    state = DummyState()


async def test_app_exception_handler_returns_json_response():
    request = DummyRequest()
    response = await app_exception_handler(request, ValueError("boom"))

    assert isinstance(response, JSONResponse)
    assert response.status_code == 500
    assert b"Internal server error" in response.body
    assert b"success" in response.body


MARKER = "student-passage-9d2e"


async def test_an_unhandled_exception_is_logged_by_type_and_place_never_by_its_text(caplog):
    try:
        raise RuntimeError(f"provider said: {MARKER}")
    except RuntimeError as exc:
        raised = exc

    with caplog.at_level(logging.ERROR):
        response = await app_exception_handler(DummyRequest(), raised)

    assert response.status_code == 500
    assert MARKER.encode() not in response.body
    record = next(r for r in caplog.records if "Unhandled exception" in r.getMessage())
    assert MARKER not in record.getMessage()
    assert record.exc_info is None
    assert "RuntimeError" in record.getMessage()
    assert "test_core_errors.py" in record.getMessage()
    assert "test-request-id" in record.getMessage()
