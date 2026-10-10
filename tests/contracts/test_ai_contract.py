"""M5-1: every signed /v1/ai route publishes a typed contract in OpenAPI.

The request body and the response `data` are named models, never `Any`, so
the backend (and any later consumer) can rely on the document.
"""

import pytest

from app.main import create_app

SPEC = create_app().openapi()
AI_PATHS = sorted(p for p in SPEC["paths"] if p.startswith("/v1/ai/"))
EXPECTED = {
    "/v1/ai/chat",
    "/v1/ai/explain",
    "/v1/ai/generate-questions",
    "/v1/ai/memory/query",
    "/v1/ai/memory/upsert",
    "/v1/ai/read/start",
    "/v1/ai/retrieve",
    "/v1/ai/tree/generate",
}


def schema(ref: dict) -> dict:
    name = ref["$ref"].rsplit("/", 1)[-1]
    return SPEC["components"]["schemas"][name]


def test_the_ai_routes_are_exactly_the_contracted_ones():
    assert set(AI_PATHS) == EXPECTED


@pytest.mark.parametrize("path", AI_PATHS)
def test_request_and_response_data_are_typed(path):
    operation = SPEC["paths"][path]["post"]
    request = schema(operation["requestBody"]["content"]["application/json"]["schema"])
    assert request.get("additionalProperties") is False, "request bodies forbid extra fields"
    assert not {"tenant_id", "user_id", "membership_id", "role"} & set(request["properties"])

    response = schema(operation["responses"]["200"]["content"]["application/json"]["schema"])
    data = response["properties"]["data"]
    # A named model, or a list of one; never an untyped `Any`.
    assert "$ref" in data or "$ref" in data.get("items", {}), data
    assert "data" in response.get("required", [])
