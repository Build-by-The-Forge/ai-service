"""Signed responses (M3-D provenance): the shared test vector, and what the
signature binds. The backend implements verification separately; this
vector, not shared code, proves both sides compute the same signature."""

import hashlib

import pytest

from app.core.service_signature_v1 import (
    AI_SERVICE_ID,
    RESPONSE_SIGNATURE_HEADER,
    RESPONSE_TIMESTAMP_HEADER,
    CanonicalResponse,
    ResponseSigner,
)

# ---- The response test vector --------------------------------------------------------
RESPONSE_VECTOR = {
    "secret": "test-vector-ai-to-backend-secret-0123456789",
    "sender": "ai-service",
    "request_id": "4f1c2a7e-4b3d-4c8e-9a1f-2d3e4f5a6b7c",
    "nonce": "Zt3m0VbqO9aX1yR4kLw2pQ8sHn6uJc5d",
    "timestamp": "1767225600",
    "body": (
        b'{"success":true,"message":"Chat response generated","sources_used":'
        b'[{"asset_id":"a1","run_number":1,"attempt":2}],"data":{}}'
    ),
    "body_sha256": "a91ea3da217dda3d1ddee30233d2849b3510acb173f6799298b6b55476508468",
    "canonical": (
        "v1\nresponse\nai-service\n4f1c2a7e-4b3d-4c8e-9a1f-2d3e4f5a6b7c\n"
        "Zt3m0VbqO9aX1yR4kLw2pQ8sHn6uJc5d\n1767225600\n"
        "a91ea3da217dda3d1ddee30233d2849b3510acb173f6799298b6b55476508468"
    ),
    "signature": "v1=8beebb6dbbcfd7168ed73ea5c5ca74b5da85717128ea028974a7bdb98c4d3e7d",
}


def vector_canonical(**changes) -> CanonicalResponse:
    fields = {
        "sender": RESPONSE_VECTOR["sender"],
        "request_id": RESPONSE_VECTOR["request_id"],
        "nonce": RESPONSE_VECTOR["nonce"],
        "timestamp": RESPONSE_VECTOR["timestamp"],
        "body_sha256": RESPONSE_VECTOR["body_sha256"],
    }
    return CanonicalResponse(**{**fields, **changes})


def test_the_vector_body_digest():
    assert hashlib.sha256(RESPONSE_VECTOR["body"]).hexdigest() == RESPONSE_VECTOR["body_sha256"]


def test_the_vector_canonical_bytes_and_signature():
    canonical = vector_canonical()
    assert canonical.to_bytes() == RESPONSE_VECTOR["canonical"].encode()
    assert canonical.signature(RESPONSE_VECTOR["secret"]) == RESPONSE_VECTOR["signature"]


def test_the_signer_reproduces_the_vector():
    signer = ResponseSigner(sender=AI_SERVICE_ID, secret=RESPONSE_VECTOR["secret"])
    headers = signer.headers(
        request_id=RESPONSE_VECTOR["request_id"],
        nonce=RESPONSE_VECTOR["nonce"],
        body=RESPONSE_VECTOR["body"],
        timestamp=int(RESPONSE_VECTOR["timestamp"]),
    )
    assert headers == {
        RESPONSE_TIMESTAMP_HEADER: RESPONSE_VECTOR["timestamp"],
        RESPONSE_SIGNATURE_HEADER: RESPONSE_VECTOR["signature"],
    }


@pytest.mark.parametrize(
    "change",
    [
        {"request_id": "9d8c7b6a-5f4e-4d3c-8b2a-1f0e9d8c7b6a"},  # replayed onto another request
        {"nonce": "Zt3m0VbqO9aX1yR4kLw2pQ8sHn6uJc5e"},  # another nonce
        {"timestamp": "1767225601"},
        {"body_sha256": hashlib.sha256(b"{}").hexdigest()},  # the body changed
        {"sender": "build-backend"},
    ],
    ids=["request", "nonce", "timestamp", "body", "sender"],
)
def test_each_bound_field_changes_the_signature(change):
    assert vector_canonical(**change).signature(RESPONSE_VECTOR["secret"]) != (
        RESPONSE_VECTOR["signature"]
    )


def test_another_secret_gives_another_signature():
    other = "test-vector-backend-to-ai-secret-9876543210"
    assert vector_canonical().signature(other) != RESPONSE_VECTOR["signature"]
