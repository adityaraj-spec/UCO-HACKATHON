"""
tests/test_enrollment_service.py

Unit tests for EnrollmentService that mock out the database session and
ECAPAService, letting us exercise business logic in complete isolation.
"""

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
import torch

from app.schemas.enrollment import EnrollmentResponse
from app.services.enrollment_service import EnrollmentService
from app.utils.exceptions import EnrollmentRejectedError, InsufficientRecordingsError, UserNotFoundError


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_fake_user(user_id: uuid.UUID | None = None) -> MagicMock:
    user = MagicMock()
    user.id = user_id or uuid.uuid4()
    user.name = "Test User"
    user.email = "test@example.com"
    return user


def _make_fake_voiceprint(user_id: uuid.UUID, embedding_dim: int = 192) -> MagicMock:
    vp = MagicMock()
    vp.id = uuid.uuid4()
    vp.user_id = user_id
    vp.embedding = [0.1] * embedding_dim
    vp.recording_count = 5
    return vp


def _make_upload_file(name: str = "test.wav") -> MagicMock:
    uf = MagicMock()
    uf.filename = name
    uf.content_type = "audio/wav"
    return uf


def _make_liveness_mock(ai_probability: float = 0.05) -> MagicMock:
    mock = MagicMock()
    mock.predict_ai_probability.return_value = ai_probability
    return mock


def _quality_patch():
    waveform = torch.cat([torch.zeros(1, 4000), torch.ones(1, 48000) * 0.2], dim=1)
    return (
        patch("app.services.enrollment_service.load_waveform", return_value=(waveform, 16000)),
        patch("app.services.enrollment_service.validate_enrollment_audio_quality"),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_enroll_raises_user_not_found_when_user_missing():
    session = AsyncMock()
    service = EnrollmentService(session=session, liveness_service=_make_liveness_mock())

    user_repo_mock = AsyncMock()
    user_repo_mock.get_by_id.return_value = None

    with patch.object(service, "user_repo", user_repo_mock):
        with pytest.raises(UserNotFoundError):
            await service.enroll(
                user_id=uuid.uuid4(),
                audio_files=[_make_upload_file()],
            )


@pytest.mark.asyncio
async def test_enroll_raises_insufficient_recordings_when_too_few_processed():
    user_id = uuid.uuid4()
    fake_user = _make_fake_user(user_id)

    session = AsyncMock()
    service = EnrollmentService(session=session, liveness_service=_make_liveness_mock())

    user_repo_mock = AsyncMock()
    user_repo_mock.get_by_id.return_value = fake_user

    ecapa_mock = MagicMock()
    # Simulate all files failing → processed_count = 0
    ecapa_mock.enroll_user.return_value = (
        np.zeros(192, dtype=np.float32), 0
    )

    load_patch, validate_patch = _quality_patch()
    with (
        patch.object(service, "user_repo", user_repo_mock),
        patch.object(service, "ecapa_service", ecapa_mock),
        load_patch,
        validate_patch,
        patch(
            "app.services.enrollment_service.save_upload_to_temp",
            new_callable=AsyncMock,
            return_value="/tmp/fake.wav",
        ),
        patch("app.services.enrollment_service.cleanup_temp_file"),
    ):
        with pytest.raises(InsufficientRecordingsError):
            await service.enroll(
                user_id=user_id,
                audio_files=[_make_upload_file()],
            )


@pytest.mark.asyncio
async def test_enroll_returns_enrollment_response_on_success():
    user_id = uuid.uuid4()
    fake_user = _make_fake_user(user_id)
    fake_vp = _make_fake_voiceprint(user_id)

    session = AsyncMock()
    service = EnrollmentService(session=session, liveness_service=_make_liveness_mock())

    user_repo_mock = AsyncMock()
    user_repo_mock.get_by_id.return_value = fake_user

    voiceprint_repo_mock = AsyncMock()
    voiceprint_repo_mock.upsert.return_value = fake_vp

    ecapa_mock = MagicMock()
    ecapa_mock.enroll_user.return_value = (
        np.random.rand(192).astype(np.float32), 5
    )

    load_patch, validate_patch = _quality_patch()
    with (
        patch.object(service, "user_repo", user_repo_mock),
        patch.object(service, "voiceprint_repo", voiceprint_repo_mock),
        patch.object(service, "ecapa_service", ecapa_mock),
        load_patch,
        validate_patch,
        patch(
            "app.services.enrollment_service.save_upload_to_temp",
            new_callable=AsyncMock,
            return_value="/tmp/fake.wav",
        ),
        patch("app.services.enrollment_service.cleanup_temp_file"),
    ):
        response = await service.enroll(
            user_id=user_id,
            audio_files=[_make_upload_file() for _ in range(5)],
        )

    assert isinstance(response, EnrollmentResponse)
    assert response.success is True
    assert response.user_id == user_id
    assert response.recording_count == 5
    assert response.embedding_dimension == 192


@pytest.mark.asyncio
async def test_enroll_rejects_when_layer1_flags_synthetic_audio(monkeypatch):
    user_id = uuid.uuid4()
    fake_user = _make_fake_user(user_id)

    monkeypatch.setattr("app.services.enrollment_service.settings.LAYER1_LIVENESS_CHECK_ENABLED", True)

    session = AsyncMock()
    service = EnrollmentService(session=session, liveness_service=_make_liveness_mock(ai_probability=0.95))

    user_repo_mock = AsyncMock()
    user_repo_mock.get_by_id.return_value = fake_user

    load_patch, validate_patch = _quality_patch()
    with (
        patch.object(service, "user_repo", user_repo_mock),
        load_patch,
        validate_patch,
        patch(
            "app.services.enrollment_service.save_upload_to_temp",
            new_callable=AsyncMock,
            return_value="/tmp/fake.wav",
        ),
        patch("app.services.enrollment_service.cleanup_temp_file"),
    ):
        with pytest.raises(EnrollmentRejectedError):
            await service.enroll(
                user_id=user_id,
                audio_files=[_make_upload_file()],
            )

    session.rollback.assert_called_once()


@pytest.mark.asyncio
async def test_enroll_dev_bypass_skips_layer1_liveness_check(monkeypatch):
    user_id = uuid.uuid4()
    fake_user = _make_fake_user(user_id)
    fake_vp = _make_fake_voiceprint(user_id)

    monkeypatch.setattr("app.services.enrollment_service.settings.LAYER1_LIVENESS_CHECK_ENABLED", False)

    session = AsyncMock()
    liveness_mock = _make_liveness_mock(ai_probability=0.95)
    service = EnrollmentService(session=session, liveness_service=liveness_mock)

    user_repo_mock = AsyncMock()
    user_repo_mock.get_by_id.return_value = fake_user

    voiceprint_repo_mock = AsyncMock()
    voiceprint_repo_mock.upsert.return_value = fake_vp

    ecapa_mock = MagicMock()
    ecapa_mock.enroll_user.return_value = (
        np.random.rand(192).astype(np.float32), 5
    )

    load_patch, validate_patch = _quality_patch()
    with (
        patch.object(service, "user_repo", user_repo_mock),
        patch.object(service, "voiceprint_repo", voiceprint_repo_mock),
        patch.object(service, "ecapa_service", ecapa_mock),
        load_patch,
        validate_patch,
        patch(
            "app.services.enrollment_service.save_upload_to_temp",
            new_callable=AsyncMock,
            return_value="/tmp/fake.wav",
        ),
        patch("app.services.enrollment_service.cleanup_temp_file"),
    ):
        response = await service.enroll(
            user_id=user_id,
            audio_files=[_make_upload_file() for _ in range(5)],
        )

    assert response.success is True
    liveness_mock.predict_ai_probability.assert_not_called()


@pytest.mark.asyncio
async def test_enroll_rolls_back_and_reraises_on_unexpected_error():
    user_id = uuid.uuid4()
    fake_user = _make_fake_user(user_id)

    session = AsyncMock()
    service = EnrollmentService(session=session, liveness_service=_make_liveness_mock())

    user_repo_mock = AsyncMock()
    user_repo_mock.get_by_id.return_value = fake_user

    ecapa_mock = MagicMock()
    ecapa_mock.enroll_user.side_effect = RuntimeError("GPU out of memory")

    load_patch, validate_patch = _quality_patch()
    with (
        patch.object(service, "user_repo", user_repo_mock),
        patch.object(service, "ecapa_service", ecapa_mock),
        load_patch,
        validate_patch,
        patch(
            "app.services.enrollment_service.save_upload_to_temp",
            new_callable=AsyncMock,
            return_value="/tmp/fake.wav",
        ),
        patch("app.services.enrollment_service.cleanup_temp_file"),
    ):
        with pytest.raises(RuntimeError, match="GPU out of memory"):
            await service.enroll(
                user_id=user_id,
                audio_files=[_make_upload_file()],
            )

    session.rollback.assert_called_once()


@pytest.mark.asyncio
async def test_reenrollment_from_unrecognized_device_requires_otp():
    user_id = uuid.uuid4()
    fake_user = _make_fake_user(user_id)
    fake_existing_vp = _make_fake_voiceprint(user_id)

    session = AsyncMock()
    service = EnrollmentService(session=session, liveness_service=_make_liveness_mock())

    user_repo_mock = AsyncMock()
    user_repo_mock.get_by_id.return_value = fake_user

    voiceprint_repo_mock = AsyncMock()
    voiceprint_repo_mock.get_by_user_id.return_value = fake_existing_vp

    with (
        patch.object(service, "user_repo", user_repo_mock),
        patch.object(service, "voiceprint_repo", voiceprint_repo_mock),
        patch("app.services.enrollment_access.redis_service.get", return_value=None),
        patch("app.services.enrollment_service.save_upload_to_temp", new_callable=AsyncMock) as save_mock,
    ):
        with pytest.raises(EnrollmentRejectedError, match="unrecognized device"):
            await service.enroll(
                user_id=user_id,
                audio_files=[_make_upload_file()],
                device_id="new-device",
                otp_verified=False,
            )

    save_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_first_time_enrollment_from_new_device_is_unaffected():
    user_id = uuid.uuid4()
    fake_user = _make_fake_user(user_id)
    fake_vp = _make_fake_voiceprint(user_id)

    session = AsyncMock()
    service = EnrollmentService(session=session, liveness_service=_make_liveness_mock())

    user_repo_mock = AsyncMock()
    user_repo_mock.get_by_id.return_value = fake_user

    voiceprint_repo_mock = AsyncMock()
    voiceprint_repo_mock.get_by_user_id.return_value = None
    voiceprint_repo_mock.upsert.return_value = fake_vp

    ecapa_mock = MagicMock()
    ecapa_mock.enroll_user.return_value = (
        np.random.rand(192).astype(np.float32), 5
    )

    load_patch, validate_patch = _quality_patch()
    with (
        patch.object(service, "user_repo", user_repo_mock),
        patch.object(service, "voiceprint_repo", voiceprint_repo_mock),
        patch.object(service, "ecapa_service", ecapa_mock),
        load_patch,
        validate_patch,
        patch(
            "app.services.enrollment_service.save_upload_to_temp",
            new_callable=AsyncMock,
            return_value="/tmp/fake.wav",
        ),
        patch("app.services.enrollment_service.cleanup_temp_file"),
        patch("app.services.enrollment_access.redis_service.set", return_value=True) as remember_mock,
    ):
        response = await service.enroll(
            user_id=user_id,
            audio_files=[_make_upload_file() for _ in range(5)],
            device_id="first-device",
        )

    assert response.success is True
    remember_mock.assert_any_call(
        f"enrollment_device:{user_id}:first-device", "known", ex=31536000
    )


@pytest.mark.asyncio
async def test_secure_mailer_enrollment_requires_login_and_otp_before_recording():
    user_id = uuid.uuid4()
    fake_user = _make_fake_user(user_id)

    session = AsyncMock()
    service = EnrollmentService(session=session, liveness_service=_make_liveness_mock())

    user_repo_mock = AsyncMock()
    user_repo_mock.get_by_id.return_value = fake_user

    voiceprint_repo_mock = AsyncMock()
    voiceprint_repo_mock.get_by_user_id.return_value = None

    with (
        patch.object(service, "user_repo", user_repo_mock),
        patch.object(service, "voiceprint_repo", voiceprint_repo_mock),
        patch("app.services.enrollment_service.save_upload_to_temp", new_callable=AsyncMock) as save_mock,
    ):
        with pytest.raises(EnrollmentRejectedError, match="login and OTP"):
            await service.enroll(
                user_id=user_id,
                audio_files=[_make_upload_file()],
                channel="SECURE_MAILER",
                biometric_consent_confirmed=True,
                authenticated=False,
                otp_verified=False,
            )

    save_mock.assert_not_awaited()
