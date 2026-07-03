"""
tests/test_enrollment_service.py

Unit tests for EnrollmentService that mock out the database session and
ECAPAService, letting us exercise business logic in complete isolation.
"""

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from app.schemas.enrollment import EnrollmentResponse
from app.services.enrollment_service import EnrollmentService
from app.utils.exceptions import InsufficientRecordingsError, UserNotFoundError


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


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_enroll_raises_user_not_found_when_user_missing():
    session = AsyncMock()
    service = EnrollmentService(session=session)

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
    service = EnrollmentService(session=session)

    user_repo_mock = AsyncMock()
    user_repo_mock.get_by_id.return_value = fake_user

    ecapa_mock = MagicMock()
    # Simulate extraction failure (e.g. returns or raises) by mocking process_file to fail
    # or ecapa extract_embedding to fail/return None
    ecapa_mock.extract_embedding.side_effect = RuntimeError("Extraction failed")

    with (
        patch.object(service, "user_repo", user_repo_mock),
        patch.object(service, "ecapa", ecapa_mock),
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

    session = AsyncMock()
    service = EnrollmentService(session=session)

    user_repo_mock = AsyncMock()
    user_repo_mock.get_by_id.return_value = fake_user

    anchor_repo_mock = AsyncMock()
    threshold_repo_mock = AsyncMock()
    threshold_repo_mock.get_or_create.return_value = MagicMock()

    ecapa_mock = MagicMock()
    ecapa_mock.extract_embedding.return_value = np.random.rand(192).astype(np.float32)

    vault_mock = MagicMock()
    vault_mock.generate_template.return_value = (b"enc_blob", b"nonce", "val_hash", "key_used")
    vault_mock.decrypt_template.return_value = np.zeros(192, dtype=np.float32)

    audio_pipeline_mock = MagicMock()
    proc_result = MagicMock()
    proc_result.accepted = True
    proc_result.waveform = np.zeros(16000, dtype=np.float32)
    proc_result.sample_rate = 16000
    audio_pipeline_mock.process_file.return_value = proc_result

    with (
        patch.object(service, "user_repo", user_repo_mock),
        patch.object(service, "anchor_repo", anchor_repo_mock),
        patch.object(service, "threshold_repo", threshold_repo_mock),
        patch.object(service, "vault", vault_mock),
        patch.object(service, "ecapa", ecapa_mock),
        patch.object(service, "audio_pipeline", audio_pipeline_mock),
        patch(
            "app.services.enrollment_service.save_upload_to_temp",
            new_callable=AsyncMock,
            return_value="/tmp/fake.wav",
        ),
        patch("app.services.enrollment_service.cleanup_temp_file"),
        patch("soundfile.write"),
        patch("app.search.faiss_index.get_faiss_manager", return_value=MagicMock()),
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
async def test_enroll_rolls_back_and_reraises_on_unexpected_error():
    user_id = uuid.uuid4()
    fake_user = _make_fake_user(user_id)

    session = AsyncMock()
    service = EnrollmentService(session=session)

    user_repo_mock = AsyncMock()
    user_repo_mock.get_by_id.return_value = fake_user

    ecapa_mock = MagicMock()
    ecapa_mock.extract_embedding.side_effect = RuntimeError("GPU out of memory")

    audio_pipeline_mock2 = MagicMock()
    proc_result2 = MagicMock()
    proc_result2.accepted = True
    proc_result2.waveform = np.zeros(16000, dtype=np.float32)
    proc_result2.sample_rate = 16000
    audio_pipeline_mock2.process_file.return_value = proc_result2

    with (
        patch.object(service, "user_repo", user_repo_mock),
        patch.object(service, "ecapa", ecapa_mock),
        patch.object(service, "audio_pipeline", audio_pipeline_mock2),
        patch(
            "app.services.enrollment_service.save_upload_to_temp",
            new_callable=AsyncMock,
            return_value="/tmp/fake.wav",
        ),
        patch("app.services.enrollment_service.cleanup_temp_file"),
        patch("soundfile.write"),
    ):
        with pytest.raises(RuntimeError, match="GPU out of memory"):
            await service.enroll(
                user_id=user_id,
                audio_files=[_make_upload_file()],
            )

    session.rollback.assert_called_once()
