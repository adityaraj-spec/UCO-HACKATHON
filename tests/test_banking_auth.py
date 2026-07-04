"""Focused tests for the banking identity authentication primitives."""

from __future__ import annotations

import jwt
import pytest
from unittest.mock import AsyncMock, MagicMock
from starlette.requests import Request

from app.core.config import get_settings
from app.middleware.jwt_auth import create_access_token
from app.schemas.user import MPINLoginRequest, mask_phone
from app.utils.mpin_hasher import hash_mpin, verify_mpin
from app.api.v1.endpoints.users import _reject_replayed_login


def _test_request(path: str = "/") -> Request:
    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": path,
            "headers": [],
            "client": ("127.0.0.1", 12345),
        }
    )


def test_mpin_hash_utility():
    hashed = hash_mpin("123456")

    assert hashed != "123456"
    assert verify_mpin("123456", hashed) is True
    assert verify_mpin("654321", hashed) is False


def test_jwt_contains_banking_claims():
    import uuid

    user_id = uuid.uuid4()
    token = create_access_token(
        user_id=user_id,
        account_number="UCO0012345678",
        full_name="Test User",
    )

    claims = jwt.decode(
        token,
        get_settings().JWT_SECRET_KEY,
        algorithms=[get_settings().JWT_ALGORITHM],
    )
    assert claims["sub"] == str(user_id)
    assert claims["account_number"] == "UCO0012345678"
    assert claims["full_name"] == "Test User"
    assert claims["role"] == "CUSTOMER"
    assert "jti" in claims


def test_mask_phone_last_four_only():
    assert mask_phone("9876543210") == "XXXXXX3210"


@pytest.mark.asyncio
async def test_login_rejects_replayed_request():
    payload = MPINLoginRequest(
        account_number="UCO0012345678",
        mpin="123456",
        request_nonce="same-nonce",
    )

    await _reject_replayed_login(payload)
    with pytest.raises(Exception) as exc:
        await _reject_replayed_login(payload)

    assert getattr(exc.value, "status_code", None) == 401


@pytest.mark.asyncio
async def test_enrollment_service_requires_active_consent():
    import uuid
    from types import SimpleNamespace

    from app.services.enrollment_service import EnrollmentService

    service = EnrollmentService(AsyncMock())
    service.user_repo.get_by_id = AsyncMock(
        return_value=SimpleNamespace(
            id=uuid.uuid4(),
            is_active=True,
            kyc_status="VERIFIED",
        )
    )
    service.consent_service.verify_active_consent = AsyncMock(return_value=False)

    with pytest.raises(PermissionError) as exc:
        await service.enroll(uuid.uuid4(), [MagicMock()])

    assert "consent" in str(exc.value).lower()


@pytest.mark.asyncio
async def test_enrollment_service_blocks_unverified_kyc_before_audio_processing():
    import uuid
    from types import SimpleNamespace

    from app.services.enrollment_service import EnrollmentService

    service = EnrollmentService(AsyncMock())
    service.user_repo.get_by_id = AsyncMock(
        return_value=SimpleNamespace(
            id=uuid.uuid4(),
            is_active=True,
            kyc_status="PENDING",
        )
    )
    service.consent_service.verify_active_consent = AsyncMock(return_value=True)

    with pytest.raises(PermissionError) as exc:
        await service.enroll(uuid.uuid4(), [MagicMock()])

    assert "kyc" in str(exc.value).lower()


@pytest.mark.asyncio
async def test_challenge_route_registers_jti_for_single_use():
    import uuid

    from app.api.v1.endpoints.voice_auth import get_challenge
    from app.cache.session_cache import SessionCache

    db = AsyncMock()
    db_result = MagicMock()
    db_result.scalars.return_value.first.return_value = None
    db.execute = AsyncMock(return_value=db_result)

    payload = await get_challenge(
        request=_test_request("/api/v1/voice/challenge"),
        session_id="session-123",
        user_id=uuid.uuid4(),
        db=db,
    )

    claims = jwt.decode(
        payload["challenge_token"],
        get_settings().JWT_SECRET_KEY,
        algorithms=[get_settings().JWT_ALGORITHM],
    )
    assert await SessionCache().verify_and_consume_jti(claims["jti"]) is True
    assert await SessionCache().verify_and_consume_jti(claims["jti"]) is False


def test_internal_service_token_rejects_default_secret(monkeypatch):
    from fastapi import HTTPException
    from app.core.config import get_settings
    from app.middleware.rbac import require_internal_service_token

    settings = get_settings()
    monkeypatch.setattr(settings, "INTERNAL_SERVICE_TOKEN", "CHANGE_ME_INTERNAL_SERVICE_TOKEN")

    with pytest.raises(HTTPException) as exc:
        require_internal_service_token("CHANGE_ME_INTERNAL_SERVICE_TOKEN")

    assert exc.value.status_code == 503
