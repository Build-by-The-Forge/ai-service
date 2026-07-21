import pytest

from app.auth.token import create_jwt, decode_jwt
from app.core.rbac import RBAC


def test_create_and_decode_jwt_round_trip():
    token = create_jwt(subject="user-1", school_id=7, role="student")
    payload = decode_jwt(token)

    assert payload["sub"] == "user-1"
    assert payload["school_id"] == 7
    assert payload["role"] == "student"


def test_rbac_permissions():
    rbac = RBAC()

    assert rbac.can_create_exam("teacher") is True
    assert rbac.can_create_exam("student") is False
    assert rbac.can_take_exam("student") is True
    assert rbac.can_manage_users("admin") is True
    assert rbac.can_manage_users("teacher") is False


def test_decode_jwt_raises_for_invalid_token():
    with pytest.raises(ValueError, match="Invalid JWT token"):
        decode_jwt("invalid.token.value")
