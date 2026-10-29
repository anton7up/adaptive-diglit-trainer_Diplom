from __future__ import annotations

import jwt
import pytest

from app.security import create_access_token, create_refresh_token, decode_token, hash_password, verify_password


def test_password_hash_is_salted():
    first = hash_password("secret123")
    second = hash_password("secret123")
    assert first != second
    assert first.startswith("pbkdf2_sha256$")
    assert second.startswith("pbkdf2_sha256$")


def test_password_verification_accepts_correct_value():
    encoded = hash_password("correct horse battery staple")
    assert verify_password("correct horse battery staple", encoded)


def test_password_verification_rejects_wrong_value():
    encoded = hash_password("original")
    assert not verify_password("different", encoded)


def test_password_verification_rejects_legacy_sha256_string():
    assert not verify_password("1234", "03ac674216f3e15c761ee1a5e255f067953623c8b388b4459e13f978d7c846f4")


def test_access_token_roundtrip():
    token = create_access_token(17, "student", "tester")
    payload = decode_token(token, "access")
    assert payload["sub"] == "17"
    assert payload["role"] == "student"
    assert payload["username"] == "tester"
    assert payload["type"] == "access"
    assert payload["jti"]


def test_refresh_token_roundtrip():
    token, jti, expires = create_refresh_token(3, "teacher", "teach")
    payload = decode_token(token, "refresh")
    assert payload["sub"] == "3"
    assert payload["jti"] == jti
    assert payload["type"] == "refresh"
    assert expires is not None


def test_access_token_cannot_be_used_as_refresh():
    token = create_access_token(1, "student", "x")
    with pytest.raises(jwt.InvalidTokenError):
        decode_token(token, "refresh")
