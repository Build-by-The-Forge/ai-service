"""Service context contract (§A.4) and exact permission membership.

Covers the C10 (malformed context refused) and C14 (context only narrows)
semantics before any route uses them.
"""

import copy

import pytest

from app.core.service_context import (
    PermissionNotGrantedError,
    ServiceContext,
    ServiceContextError,
    parse_service_context,
    require_permission,
)
from app.core.service_signature_v1 import (
    BACKEND_SERVICE_ID,
    CONTEXT_HEADER,
    REQUEST_ID_HEADER,
    SIGNATURE_HEADER,
    CanonicalRequest,
    ServiceSigner,
    ServiceVerifier,
    SignatureVerificationError,
    context_to_header,
    hex_sha256,
    serialize_context,
)

TENANT = "6f1c2a7e-4b3d-4c8e-9a1f-2d3e4f5a6b7c"
USER = "0b9d8c7a-6e5f-4a3b-8c2d-1e0f9a8b7c6d"
MEMBERSHIP = "a1b2c3d4-e5f6-4a7b-8c9d-0e1f2a3b4c5d"

USER_CTX = {
    "actor": "user",
    "correlation_id": "corr-001",
    "membership_id": MEMBERSHIP,
    "permissions": ["ai.questions.generate"],
    "role": "lecturer",
    "tenant_id": TENANT,
    "user_id": USER,
    "v": 1,
}
SYSTEM_CTX = {
    "actor": "system",
    "correlation_id": "job-7:attempt-2",
    "membership_id": None,
    "permissions": [],
    "role": None,
    "tenant_id": TENANT,
    "user_id": None,
    "v": 1,
}


def variant(base, **changes):
    data = copy.deepcopy(base)
    data.update(changes)
    return data


def user_with(permissions, role="student"):
    return parse_service_context(variant(USER_CTX, permissions=permissions, role=role))


# ------------------------------------------------------------- valid shapes


def test_user_context_parses():
    ctx = parse_service_context(USER_CTX)
    assert (ctx.actor, ctx.tenant_id, ctx.user_id, ctx.membership_id, ctx.role) == (
        "user",
        TENANT,
        USER,
        MEMBERSHIP,
        "lecturer",
    )
    assert ctx.permission_list == ("ai.questions.generate",)
    assert not ctx.is_system
    assert ctx.to_json_object() == USER_CTX


def test_system_context_parses():
    ctx = parse_service_context(SYSTEM_CTX)
    assert ctx.is_system
    assert (ctx.user_id, ctx.membership_id, ctx.role, ctx.permission_list) == (None, None, None, ())
    assert ctx.to_json_object() == SYSTEM_CTX


def test_system_constructor_builds_the_callback_context():
    ctx = ServiceContext.system(tenant_id=TENANT, correlation_id="job-7:attempt-2")
    assert ctx.to_json_object() == SYSTEM_CTX


def test_system_constructor_validates_its_input():
    with pytest.raises(ServiceContextError):
        ServiceContext.system(tenant_id="school-1", correlation_id="job-7")


def test_multiple_sorted_permissions_are_accepted():
    ctx = user_with(["ai.questions.generate", "ai.use"])
    assert ctx.permission_list == ("ai.questions.generate", "ai.use")


# -------------------------------------------------------------- refusals


@pytest.mark.parametrize("key", sorted(USER_CTX))
def test_missing_key_is_refused(key):
    data = variant(USER_CTX)
    del data[key]
    with pytest.raises(ServiceContextError, match="missing"):
        parse_service_context(data)


@pytest.mark.parametrize("key", ["school_id", "scopes", "is_platform_admin", "Tenant_id"])
def test_unexpected_key_is_refused(key):
    with pytest.raises(ServiceContextError, match="unexpected"):
        parse_service_context(variant(USER_CTX, **{key: "x"}))


WRONG_USER_VALUES = {
    "v": [0, 2, "1", True, 1.0, None, [1]],
    "actor": ["", "USER", "service", "admin", None, 0],
    "tenant_id": [
        TENANT.upper(),
        TENANT.replace("-", ""),
        f"{{{TENANT}}}",
        f"urn:uuid:{TENANT}",
        TENANT + " ",
        "school-1",
        "",
        None,
        12,
    ],
    "user_id": [None, USER.upper(), "user-1", 3],
    "membership_id": [None, "", MEMBERSHIP[:-1]],
    "role": [None, "", 1, ["lecturer"], {"name": "lecturer"}],
    "permissions": [
        None,
        "ai.use",
        {"ai.use": True},
        ["ai.use", "ai.questions.generate"],
        ["ai.use", "ai.use"],
        [""],
        ["ai.use", None],
        [["ai.use"]],
        [1],
    ],
    "correlation_id": ["", "c" * 129, "with space", "a/b", "tab\t", "é", None, 5],
}
WRONG_USER_CASES = [(k, v) for k, values in WRONG_USER_VALUES.items() for v in values]


@pytest.mark.parametrize(
    "key,value", WRONG_USER_CASES, ids=[f"{k}-{i}" for i, (k, _) in enumerate(WRONG_USER_CASES)]
)
def test_wrong_value_is_refused(key, value):
    with pytest.raises(ServiceContextError):
        parse_service_context(variant(USER_CTX, **{key: value}))


@pytest.mark.parametrize(
    "key,value",
    [
        ("user_id", USER),
        ("membership_id", MEMBERSHIP),
        ("role", "admin"),
        ("role", ""),
        ("permissions", ["content.ingest"]),
    ],
)
def test_system_context_carries_no_user_and_no_permissions(key, value):
    with pytest.raises(ServiceContextError):
        parse_service_context(variant(SYSTEM_CTX, **{key: value}))


@pytest.mark.parametrize("value", ["c", "c" * 128, "Az09._:-"])
def test_correlation_id_limits(value):
    assert parse_service_context(variant(USER_CTX, correlation_id=value)).correlation_id == value


# ---------------------------------------- exact permission membership (C14)


def test_declared_permission_present_is_allowed():
    require_permission(user_with(["ai.questions.generate"]), "ai.questions.generate")


def test_missing_permission_is_refused():
    with pytest.raises(PermissionNotGrantedError) as raised:
        require_permission(user_with(["ai.use"]), "ai.questions.generate")
    assert raised.value.required_permission == "ai.questions.generate"


@pytest.mark.parametrize("role", ["admin", "owner", "platform_admin", "lecturer"])
def test_role_never_grants_a_permission(role):
    with pytest.raises(PermissionNotGrantedError):
        require_permission(user_with([], role=role), "ai.use")


@pytest.mark.parametrize(
    "granted,required",
    [
        (["ai.use"], "ai.questions.generate"),  # no implication between permissions
        (["ai.questions.generate"], "ai.use"),  # nor the other way round
        (["ai.*"], "ai.use"),  # no wildcards
        (["*"], "ai.use"),
        (["ai"], "ai.use"),  # no hierarchy
        (["ai.use.extended"], "ai.use"),
        (["AI.USE"], "ai.use"),  # exact case
        (["ai.use "], "ai.use"),
    ],
)
def test_membership_is_exact_string_equality(granted, required):
    with pytest.raises(PermissionNotGrantedError):
        require_permission(user_with(sorted(granted)), required)


def test_system_context_satisfies_no_permission():
    with pytest.raises(PermissionNotGrantedError):
        require_permission(parse_service_context(SYSTEM_CTX), "ai.use")


@pytest.mark.parametrize("declared", ["", None])
def test_route_must_declare_a_permission(declared):
    with pytest.raises(ValueError):
        require_permission(parse_service_context(USER_CTX), declared)


# --------------------------------------------- enforced in verification (C10)

SECRET = "test-only-backend-to-ai-secret-0123456789abcdef"
NOW = 1760000000
verifier = ServiceVerifier(accepted_sender=BACKEND_SERVICE_ID, secrets=(SECRET,))


def signed_over(context_object):
    """Correctly signed headers for any decoded context, as a sender holding
    the secret but building the context by hand could produce."""
    headers = ServiceSigner(sender=BACKEND_SERVICE_ID, secret=SECRET).headers(
        method="POST",
        target="/v1/ai/chat",
        body=b"{}",
        context=parse_service_context(USER_CTX),
        timestamp=NOW,
    )
    canonical = CanonicalRequest(
        sender=BACKEND_SERVICE_ID,
        method="POST",
        target="/v1/ai/chat",
        timestamp=str(NOW),
        request_id=headers[REQUEST_ID_HEADER],
        body_sha256=hex_sha256(b"{}"),
        context_sha256=hex_sha256(serialize_context(context_object)),
    )
    headers[CONTEXT_HEADER] = context_to_header(context_object)
    headers[SIGNATURE_HEADER] = canonical.signature(SECRET)
    return headers


def verify(headers):
    return verifier.verify(
        method="POST",
        target="/v1/ai/chat",
        headers=headers.items(),
        body_sha256=hex_sha256(b"{}"),
        now=NOW,
    )


def test_verified_call_exposes_the_parsed_context():
    assert verify(signed_over(USER_CTX)).context == parse_service_context(USER_CTX)


@pytest.mark.parametrize(
    "context_object",
    [
        variant(USER_CTX, school_id=TENANT),
        {k: v for k, v in USER_CTX.items() if k != "tenant_id"},
        variant(USER_CTX, v=True),
        variant(USER_CTX, tenant_id=TENANT.upper()),
        variant(USER_CTX, permissions=["ai.use", "ai.questions.generate"]),
        variant(SYSTEM_CTX, permissions=["ai.use"]),
    ],
    ids=["extra-key", "no-tenant", "bool-v", "upper-tenant", "unsorted-perms", "system-perms"],
)
def test_validly_signed_invalid_context_is_refused(context_object):
    with pytest.raises(SignatureVerificationError, match="context is invalid"):
        verify(signed_over(context_object))
