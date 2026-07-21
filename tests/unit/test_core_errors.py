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
