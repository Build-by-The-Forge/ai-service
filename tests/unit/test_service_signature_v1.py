"""BUILD Service Signature v1 (AI-service side).

Reproduces test vector A.7 from the M3 boundary design and checks that every
single-field change invalidates the signature (acceptance row C1), plus the
verification steps that need neither Redis nor a streamed body.
"""

import base64

import pytest

from app.core.service_context import parse_service_context
from app.core.service_signature_v1 import (
    AI_SERVICE_ID,
    BACKEND_SERVICE_ID,
    CONTEXT_HEADER,
    REQUEST_ID_HEADER,
    REQUIRED_HEADERS,
    SERVICE_HEADER,
    SHA256_OF_EMPTY,
    SIGNATURE_HEADER,
    TIMESTAMP_HEADER,
    CanonicalRequest,
    ServiceSigner,
    ServiceVerifier,
    SignatureVerificationError,
    context_to_header,
    hex_sha256,
    parse_context_header,
    serialize_context,
    wire_request_target,
)

# ---- Test vector A.7, copied verbatim from the design document ----
VECTOR = {
    "secret": "test-only-backend-to-ai-secret-0123456789abcdef",
    "service": "build-backend",
    "method": "POST",
    "target": "/v1/ai/generate-questions?lang=en",
    "timestamp": "1760000000",
    "request_id": "3f1c0d9e-6b0a-4c55-9d43-1c2b8f7a9e10",
    "body": b'{"topic":"Thermodynamics","count":5}',
    "context_json": (
        b'{"actor":"user","correlation_id":"corr-001",'
        b'"membership_id":"a1b2c3d4-e5f6-4a7b-8c9d-0e1f2a3b4c5d",'
        b'"permissions":["ai.questions.generate"],"role":"lecturer",'
        b'"tenant_id":"6f1c2a7e-4b3d-4c8e-9a1f-2d3e4f5a6b7c",'
        b'"user_id":"0b9d8c7a-6e5f-4a3b-8c2d-1e0f9a8b7c6d","v":1}'
    ),
    "context_header": (
        "eyJhY3RvciI6InVzZXIiLCJjb3JyZWxhdGlvbl9pZCI6ImNvcnItMDAxIiwibWVtYmVyc2hpcF9pZCI6"
        "ImExYjJjM2Q0LWU1ZjYtNGE3Yi04YzlkLTBlMWYyYTNiNGM1ZCIsInBlcm1pc3Npb25zIjpbImFpLnF1"
        "ZXN0aW9ucy5nZW5lcmF0ZSJdLCJyb2xlIjoibGVjdHVyZXIiLCJ0ZW5hbnRfaWQiOiI2ZjFjMmE3ZS00"
        "YjNkLTRjOGUtOWExZi0yZDNlNGY1YTZiN2MiLCJ1c2VyX2lkIjoiMGI5ZDhjN2EtNmU1Zi00YTNiLThj"
        "MmQtMWUwZjlhOGI3YzZkIiwidiI6MX0"
    ),
    "body_sha256": "7c60810327978fdb8394acfbccc7e2287850b1309330d9cb3cda5bae3c4aa70d",
    "context_sha256": "bd5bf4d02c7c63f22a40f52b497eab2ea1078fa1155956c81c111830fc2d5e06",
    "canonical": (
        "v1\nbuild-backend\nPOST\n/v1/ai/generate-questions?lang=en\n1760000000\n"
        "3f1c0d9e-6b0a-4c55-9d43-1c2b8f7a9e10\n"
        "7c60810327978fdb8394acfbccc7e2287850b1309330d9cb3cda5bae3c4aa70d\n"
        "bd5bf4d02c7c63f22a40f52b497eab2ea1078fa1155956c81c111830fc2d5e06"
    ),
    "signature": "v1=be014f19af31ec051f99db00598a19eb80bc4de18d517f408d2abaca2f719f19",
}
VECTOR_CONTEXT = {
    "actor": "user",
    "correlation_id": "corr-001",
    "membership_id": "a1b2c3d4-e5f6-4a7b-8c9d-0e1f2a3b4c5d",
    "permissions": ["ai.questions.generate"],
    "role": "lecturer",
    "tenant_id": "6f1c2a7e-4b3d-4c8e-9a1f-2d3e4f5a6b7c",
    "user_id": "0b9d8c7a-6e5f-4a3b-8c2d-1e0f9a8b7c6d",
    "v": 1,
}
VECTOR_SERVICE_CONTEXT = parse_service_context(VECTOR_CONTEXT)
NOW = 1760000000
OTHER_DIRECTION_SECRET = "test-only-ai-to-backend-secret-fedcba9876543210"

backend_signer = ServiceSigner(sender=BACKEND_SERVICE_ID, secret=VECTOR["secret"])
inbound_verifier = ServiceVerifier(accepted_sender=BACKEND_SERVICE_ID, secrets=(VECTOR["secret"],))


def vector_headers(**replace: str) -> dict[str, str]:
    headers = {
        SERVICE_HEADER: VECTOR["service"],
        TIMESTAMP_HEADER: VECTOR["timestamp"],
        REQUEST_ID_HEADER: VECTOR["request_id"],
        CONTEXT_HEADER: VECTOR["context_header"],
        SIGNATURE_HEADER: VECTOR["signature"],
    }
    headers.update(replace)
    return headers


def verify(
    headers=None,
    *,
    verifier: ServiceVerifier = inbound_verifier,
    method: str = VECTOR["method"],
    target: str = VECTOR["target"],
    body: bytes = VECTOR["body"],
    now: int = NOW,
):
    if headers is None:
        headers = vector_headers()
    pairs = list(headers.items()) if isinstance(headers, dict) else headers
    return verifier.verify(
        method=method,
        target=target,
        headers=pairs,
        body_sha256=hex_sha256(body),
        now=now,
    )


def vector_canonical(**replace: str) -> CanonicalRequest:
    fields = {
        "sender": VECTOR["service"],
        "method": VECTOR["method"],
        "target": VECTOR["target"],
        "timestamp": VECTOR["timestamp"],
        "request_id": VECTOR["request_id"],
        "body_sha256": VECTOR["body_sha256"],
        "context_sha256": VECTOR["context_sha256"],
    }
    fields.update(replace)
    return CanonicalRequest(**fields)


def b64url(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


# ---------------------------------------------------------------- C1: vector


def test_vector_context_serialisation_and_header():
    assert serialize_context(VECTOR_CONTEXT) == VECTOR["context_json"]
    assert context_to_header(VECTOR_CONTEXT) == VECTOR["context_header"]


def test_vector_hashes():
    assert hex_sha256(VECTOR["body"]) == VECTOR["body_sha256"]
    assert hex_sha256(VECTOR["context_json"]) == VECTOR["context_sha256"]


def test_vector_canonical_request_bytes():
    assert vector_canonical().to_bytes() == VECTOR["canonical"].encode("ascii")


def test_vector_signature():
    assert vector_canonical().signature(VECTOR["secret"]) == VECTOR["signature"]


def test_signer_produces_the_vector_headers():
    headers = backend_signer.headers(
        method=VECTOR["method"],
        target=VECTOR["target"],
        body=VECTOR["body"],
        context=VECTOR_SERVICE_CONTEXT,
        timestamp=NOW,
        request_id=VECTOR["request_id"],
    )
    assert headers == vector_headers()


def test_verifier_accepts_the_vector():
    call = verify()
    assert (call.sender, call.timestamp, call.request_id) == (
        BACKEND_SERVICE_ID,
        NOW,
        VECTOR["request_id"],
    )
    assert call.context == VECTOR_SERVICE_CONTEXT
    assert call.context.to_json_object() == VECTOR_CONTEXT
    assert call.raw_context == VECTOR["context_json"]


# ------------------------------------------------------ C1: negative vectors

NEGATIVE_VECTORS = [
    ("service id", {"sender": "ai-service"}),
    ("method", {"method": "GET"}),
    ("path", {"target": "/v1/ai/generate-questions/?lang=en"}),
    ("query value", {"target": "/v1/ai/generate-questions?lang=EN"}),
    ("query added", {"target": "/v1/ai/generate-questions?lang=en&tenant_id=x"}),
    ("query dropped", {"target": "/v1/ai/generate-questions"}),
    ("timestamp", {"timestamp": "1759999999"}),
    ("request id", {"request_id": "4f1c0d9e-6b0a-4c55-9d43-1c2b8f7a9e10"}),
    ("body", {"body_sha256": hex_sha256(b'{"topic":"Thermodynamics","count":50}')}),
    (
        "context",
        {"context_sha256": hex_sha256(VECTOR["context_json"].replace(b"lecturer", b"admin"))},
    ),
]


@pytest.mark.parametrize(
    "replace", [r for _, r in NEGATIVE_VECTORS], ids=[n for n, _ in NEGATIVE_VECTORS]
)
def test_changing_any_signed_field_changes_the_signature(replace):
    assert vector_canonical(**replace).signature(VECTOR["secret"]) != VECTOR["signature"]


def test_changing_the_secret_changes_the_signature():
    assert vector_canonical().signature(VECTOR["secret"][:-1] + "0") != VECTOR["signature"]


# -------------------------------- verifier refuses each tampered component


@pytest.mark.parametrize(
    "tamper",
    [
        {"method": "PATCH"},
        {"target": "/v1/ai/generate-questions/?lang=en"},
        {"target": "/v1/ai/generate-questions?lang=en&count=500"},
        {"target": "/v1/ai/generate-questions"},
        {"body": b'{"topic":"Thermodynamics","count":50}'},
        {"body": b""},
    ],
    ids=["method", "path", "query-added", "query-changed", "body", "body-removed"],
)
def test_verifier_refuses_tampered_request(tamper):
    with pytest.raises(SignatureVerificationError, match="does not match"):
        verify(**tamper)


@pytest.mark.parametrize(
    "key,value",
    [
        ("tenant_id", "11111111-1111-4111-8111-111111111111"),
        ("user_id", "22222222-2222-4222-8222-222222222222"),
        ("permissions", ["ai.questions.generate", "ai.use"]),
    ],
)
def test_verifier_refuses_swapped_context_fields(key, value):
    tampered = {**VECTOR_CONTEXT, key: value}
    with pytest.raises(SignatureVerificationError, match="does not match"):
        verify(vector_headers(**{CONTEXT_HEADER: context_to_header(tampered)}))


@pytest.mark.parametrize(
    "header,value",
    [
        (TIMESTAMP_HEADER, "1760000030"),
        (REQUEST_ID_HEADER, "4f1c0d9e-6b0a-4c55-9d43-1c2b8f7a9e10"),
    ],
)
def test_verifier_refuses_tampered_timestamp_or_request_id(header, value):
    with pytest.raises(SignatureVerificationError, match="does not match"):
        verify(vector_headers(**{header: value}))


# ------------------------------------------------- directions and rotation


def test_only_the_backend_sender_is_accepted_inbound():
    with pytest.raises(SignatureVerificationError, match="sender"):
        verify(vector_headers(**{SERVICE_HEADER: AI_SERVICE_ID}))


def test_a_verifier_for_the_other_direction_refuses_backend_requests():
    callback_verifier = ServiceVerifier(accepted_sender=AI_SERVICE_ID, secrets=(VECTOR["secret"],))
    with pytest.raises(SignatureVerificationError, match="sender"):
        verify(verifier=callback_verifier)


def test_a_request_signed_with_the_other_directions_secret_is_refused():
    wrong_key = ServiceSigner(sender=BACKEND_SERVICE_ID, secret=OTHER_DIRECTION_SECRET)
    headers = wrong_key.headers(
        method=VECTOR["method"],
        target=VECTOR["target"],
        body=VECTOR["body"],
        context=VECTOR_SERVICE_CONTEXT,
        timestamp=NOW,
    )
    with pytest.raises(SignatureVerificationError, match="does not match"):
        verify(headers)


def test_outbound_callbacks_verify_with_the_ai_to_backend_secret_only():
    signer = ServiceSigner(sender=AI_SERVICE_ID, secret=OTHER_DIRECTION_SECRET)
    headers = signer.headers(
        method="POST",
        target="/content/ingestion/callbacks",
        body=b"{}",
        context=VECTOR_SERVICE_CONTEXT,
        timestamp=NOW,
    )

    def check(secret: str):
        return ServiceVerifier(accepted_sender=AI_SERVICE_ID, secrets=(secret,)).verify(
            method="POST",
            target="/content/ingestion/callbacks",
            headers=headers.items(),
            body_sha256=hex_sha256(b"{}"),
            now=NOW,
        )

    assert check(OTHER_DIRECTION_SECRET).sender == AI_SERVICE_ID
    with pytest.raises(SignatureVerificationError, match="does not match"):
        check(VECTOR["secret"])


def test_previous_secret_is_accepted_during_rotation():
    rotated = ServiceVerifier(
        accepted_sender=BACKEND_SERVICE_ID,
        secrets=("rotated-backend-to-ai-secret-abcdefghijklmnop", VECTOR["secret"]),
    )
    assert verify(verifier=rotated).request_id == VECTOR["request_id"]

    retired = ServiceVerifier(
        accepted_sender=BACKEND_SERVICE_ID,
        secrets=("rotated-backend-to-ai-secret-abcdefghijklmnop",),
    )
    with pytest.raises(SignatureVerificationError, match="does not match"):
        verify(verifier=retired)


def test_verifier_requires_a_secret():
    with pytest.raises(ValueError):
        ServiceVerifier(accepted_sender=BACKEND_SERVICE_ID, secrets=())
    with pytest.raises(ValueError):
        ServiceVerifier(accepted_sender=BACKEND_SERVICE_ID, secrets=("",))


# ---------------------------------------------------------------- timestamp


@pytest.mark.parametrize("skew", [-60, -1, 0, 1, 60])
def test_timestamps_within_sixty_seconds_are_accepted(skew):
    verify(now=NOW + skew)


@pytest.mark.parametrize("skew", [-61, 61, 3600])
def test_timestamps_beyond_sixty_seconds_are_refused(skew):
    with pytest.raises(SignatureVerificationError, match="window"):
        verify(now=NOW + skew)


@pytest.mark.parametrize(
    "value",
    [
        "",
        "0",
        "01760000000",
        "+1760000000",
        "-1760000000",
        "1760000000.5",
        "1760000000\n",
        "١٧٦٠٠٠٠٠٠٠",
        "1760000000000",
    ],
)
def test_malformed_timestamps_are_refused(value):
    with pytest.raises(SignatureVerificationError, match="timestamp is malformed"):
        verify(vector_headers(**{TIMESTAMP_HEADER: value}))


# --------------------------------------------------------------- request id


@pytest.mark.parametrize(
    "value",
    [
        "3F1C0D9E-6B0A-4C55-9D43-1C2B8F7A9E10",
        "3f1c0d9e6b0a4c559d431c2b8f7a9e10",
        "urn:uuid:3f1c0d9e-6b0a-4c55-9d43-1c2b8f7a9e10",
        "{3f1c0d9e-6b0a-4c55-9d43-1c2b8f7a9e10}",
        "3f1c0d9e-6b0a-5c55-9d43-1c2b8f7a9e10",
        "3f1c0d9e-6b0a-4c55-cd43-1c2b8f7a9e10",
        "3f1c0d9e-6b0a-4c55-9d43-1c2b8f7a9e10\n",
        "not-a-uuid",
    ],
)
def test_request_id_must_be_canonical_lowercase_uuid4(value):
    with pytest.raises(SignatureVerificationError, match="UUIDv4"):
        verify(vector_headers(**{REQUEST_ID_HEADER: value}))


def test_signer_uses_a_new_uuid4_for_every_request():
    def sign_once():
        return backend_signer.headers(
            method="POST",
            target="/v1/ai/chat",
            body=b"{}",
            context=VECTOR_SERVICE_CONTEXT,
            timestamp=NOW,
        )

    first, second = sign_once(), sign_once()
    assert first[REQUEST_ID_HEADER] != second[REQUEST_ID_HEADER]
    for headers in (first, second):
        verify(headers, target="/v1/ai/chat", body=b"{}")


# ------------------------------------------------------------------ headers


@pytest.mark.parametrize("name", REQUIRED_HEADERS)
def test_every_header_is_required(name):
    headers = vector_headers()
    headers.pop(name)
    with pytest.raises(SignatureVerificationError, match="absent headers"):
        verify(headers)


def test_header_names_match_case_insensitively():
    verify({name.upper(): value for name, value in vector_headers().items()})


def test_a_header_sent_twice_is_refused():
    pairs = list(vector_headers().items()) + [("x-build-signature", VECTOR["signature"])]
    with pytest.raises(SignatureVerificationError, match="more than once"):
        verify(pairs)


@pytest.mark.parametrize(
    "value",
    [
        VECTOR["signature"].replace("v1=", "V1="),
        VECTOR["signature"].replace("v1=", ""),
        VECTOR["signature"].upper(),
        VECTOR["signature"][:-1],
        VECTOR["signature"] + "\n",
        "Bearer some-static-key",
    ],
)
def test_malformed_signature_headers_are_refused(value):
    with pytest.raises(SignatureVerificationError, match="signature header is malformed"):
        verify(vector_headers(**{SIGNATURE_HEADER: value}))


# ---------------------------------------------------------- context encoding


def test_context_header_parses_to_signed_bytes_and_object():
    raw, parsed = parse_context_header(VECTOR["context_header"])
    assert raw == VECTOR["context_json"]
    assert parsed == VECTOR_CONTEXT


@pytest.mark.parametrize(
    "value,detail",
    [
        (VECTOR["context_header"] + "=", "unpadded base64url"),
        ("eyJhIjoi++++In0", "unpadded base64url"),
        (VECTOR["context_header"][:-1] + "1", "non-canonical base64url"),
        (VECTOR["context_header"] + "xy", "unpadded base64url"),
        ("e30\n", "unpadded base64url"),
        ("x" * 4097, "exceeds 4096"),
    ],
    ids=["padding", "standard-alphabet", "unused-bits", "impossible-length", "newline", "oversize"],
)
def test_badly_encoded_context_headers_are_refused(value, detail):
    with pytest.raises(SignatureVerificationError, match=detail):
        parse_context_header(value)


@pytest.mark.parametrize(
    "raw,detail",
    [
        (b'{"v":1,"actor":"user"}', "not canonical"),
        (b'{"actor": "user"}', "not canonical"),
        (b'{"actor":"user"} ', "not canonical"),
        (b'{"v":1,"v":1}', "not valid ASCII JSON"),
        (b'{"v":Infinity}', "not valid ASCII JSON"),
        (b'{"role":"l\xc3\xa9cturer"}', "not valid ASCII JSON"),
        (b"", "not valid ASCII JSON"),
        (b'["actor"]', "must be a JSON object"),
        (b'"actor"', "must be a JSON object"),
    ],
    ids=[
        "unsorted",
        "spaces",
        "trailing-space",
        "duplicate",
        "infinity",
        "raw-utf8",
        "empty",
        "array",
        "string",
    ],
)
def test_non_canonical_context_json_is_refused(raw, detail):
    with pytest.raises(SignatureVerificationError, match=detail):
        parse_context_header(b64url(raw))


def test_unicode_in_context_is_escaped_to_ascii():
    raw = serialize_context({"role": "Ọ̀jọ́gbọ́n"})
    assert raw.isascii()
    assert parse_context_header(b64url(raw))[1] == {"role": "Ọ̀jọ́gbọ́n"}


def test_valid_signature_over_non_canonical_context_is_still_refused():
    raw = b'{"v": 1}'
    canonical = vector_canonical(context_sha256=hex_sha256(raw))
    headers = vector_headers(
        **{CONTEXT_HEADER: b64url(raw), SIGNATURE_HEADER: canonical.signature(VECTOR["secret"])}
    )
    with pytest.raises(SignatureVerificationError, match="not canonical"):
        verify(headers)


# ------------------------------------------------ request target and body


def test_wire_request_target_is_not_decoded_or_reordered():
    assert (
        wire_request_target(b"/v1/ai/tree/a%2Fb%20c", b"z=1&a=2") == "/v1/ai/tree/a%2Fb%20c?z=1&a=2"
    )
    assert wire_request_target(b"/v1/ai/chat", b"") == "/v1/ai/chat"


def test_wire_request_target_rejects_non_ascii():
    with pytest.raises(SignatureVerificationError):
        wire_request_target(b"/v1/ai/chat", "q=é".encode())


def test_decoded_path_does_not_match_the_signed_encoded_path():
    headers = backend_signer.headers(
        method="GET",
        target="/v1/ai/read/a%2Fb",
        body=b"",
        context=VECTOR_SERVICE_CONTEXT,
        timestamp=NOW,
    )
    verify(headers, method="GET", target="/v1/ai/read/a%2Fb", body=b"")
    with pytest.raises(SignatureVerificationError, match="does not match"):
        verify(headers, method="GET", target="/v1/ai/read/a/b", body=b"")


def test_empty_body_digest_constant():
    assert hex_sha256(b"") == SHA256_OF_EMPTY
