"""
tests/test_verification_service.py

Unit tests for VerificationService that mock out the database session,
ECAPA service, and file utilities.
"""

import uuid
from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
import torch

from app.schemas.verification import VerificationResult
from app.services.verification_service import (
    DECISION_MISMATCH,
    DECISION_STEP_UP,
    DECISION_VERIFIED,
    VerificationService,
)
from app.services.transaction_policy import DECISION_OTP_REQUIRED, DECISION_PHRASE_MISMATCH
from app.utils.exceptions import UserNotFoundError, VoiceprintNotFoundError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fake_user(user_id: uuid.UUID | None = None) -> MagicMock:
    user = MagicMock()
    user.id = user_id or uuid.uuid4()
    user.name = "Test User"
    return user


def _make_fake_voiceprint(user_id: uuid.UUID, embedding_dim: int = 192) -> MagicMock:
    vp = MagicMock()
    vp.id = uuid.uuid4()
    vp.user_id = user_id
    embedding = np.linspace(-1.0, 1.0, embedding_dim, dtype=np.float32)
    embedding = embedding / np.linalg.norm(embedding)
    vp.embedding = embedding.tolist()
    vp.biohash = None
    vp.model_version = "test-model"
    return vp


def _make_upload_file(name: str = "live.wav") -> MagicMock:
    uf = MagicMock()
    uf.filename = name
    uf.content_type = "audio/wav"
    return uf


@contextmanager
def _mock_current_verification_pipeline(
    service: VerificationService,
    user_repo_mock: AsyncMock,
    voiceprint_repo_mock: AsyncMock,
    ver_log_repo_mock: AsyncMock,
    risk_log_repo_mock: AsyncMock,
    ecapa_mock: MagicMock,
    snorm_mock: MagicMock,
):
    waveform = torch.zeros(1, 16000 * 4)

    with (
        patch.object(service, "user_repo", user_repo_mock),
        patch.object(service, "voiceprint_repo", voiceprint_repo_mock),
        patch.object(service, "verification_log_repo", ver_log_repo_mock),
        patch.object(service, "risk_log_repo", risk_log_repo_mock),
        patch.object(service, "ecapa_service", ecapa_mock),
        patch.object(service, "snorm_service", snorm_mock),
        patch(
            "app.services.verification_service.save_upload_to_temp",
            new_callable=AsyncMock,
            return_value="/tmp/live.wav",
        ),
        patch("app.services.verification_service.load_waveform", return_value=(waveform, 16000)),
        patch(
            "app.services.verification_service.diarise_and_extract_dominant_speaker",
            return_value=(waveform, 4.0),
        ),
        patch("app.services.verification_service.compute_perceptual_audio_hash", return_value="audio-hash"),
        patch(
            "app.services.verification_service.redis_service.check_and_store_audio_hash",
            return_value=True,
        ),
        patch("app.services.verification_service.cleanup_temp_file"),
    ):
        yield


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_verify_raises_user_not_found():
    session = AsyncMock()
    session.add = MagicMock()
    service = VerificationService(session=session)

    user_repo_mock = AsyncMock()
    user_repo_mock.get_by_id.return_value = None

    with patch.object(service, "user_repo", user_repo_mock):
        with pytest.raises(UserNotFoundError):
            await service.verify(
                user_id=uuid.uuid4(),
                audio_file=_make_upload_file(),
            )


@pytest.mark.asyncio
async def test_verify_raises_voiceprint_not_found():
    user_id = uuid.uuid4()
    session = AsyncMock()
    session.add = MagicMock()
    service = VerificationService(session=session)

    user_repo_mock = AsyncMock()
    user_repo_mock.get_by_id.return_value = _make_fake_user(user_id)

    voiceprint_repo_mock = AsyncMock()
    voiceprint_repo_mock.get_by_user_id.return_value = None

    with (
        patch.object(service, "user_repo", user_repo_mock),
        patch.object(service, "voiceprint_repo", voiceprint_repo_mock),
    ):
        with pytest.raises(VoiceprintNotFoundError):
            await service.verify(
                user_id=user_id,
                audio_file=_make_upload_file(),
            )


@pytest.mark.asyncio
async def test_verify_returns_verified_when_snorm_passes():
    user_id = uuid.uuid4()
    fake_vp = _make_fake_voiceprint(user_id)
    live_embedding = np.asarray(fake_vp.embedding, dtype=np.float32)

    session = AsyncMock()
    session.add = MagicMock()
    service = VerificationService(session=session)

    user_repo_mock = AsyncMock()
    user_repo_mock.get_by_id.return_value = _make_fake_user(user_id)

    voiceprint_repo_mock = AsyncMock()
    voiceprint_repo_mock.get_by_user_id.return_value = fake_vp

    ver_log_repo_mock = AsyncMock()
    risk_log_repo_mock = AsyncMock()

    ecapa_mock = MagicMock()
    ecapa_mock.extract_embedding_with_confidence.return_value = (
        live_embedding,
        1.0,
        1.0,
    )

    snorm_mock = MagicMock()
    snorm_mock.compute_snorm_score = AsyncMock(return_value=(3.5, 0.0, 0.2))
    snorm_mock.evaluate_decision.return_value = (DECISION_VERIFIED, True)

    with _mock_current_verification_pipeline(
        service,
        user_repo_mock,
        voiceprint_repo_mock,
        ver_log_repo_mock,
        risk_log_repo_mock,
        ecapa_mock,
        snorm_mock,
    ):
        result = await service.verify(
            user_id=user_id,
            audio_file=_make_upload_file(),
            layer1_score=0.05,
        )

    assert isinstance(result, VerificationResult)
    assert result.verified is True
    assert result.decision == DECISION_VERIFIED
    assert result.similarity_score == pytest.approx(1.0)
    assert result.risk_level == "CLEAN"
    snorm_mock.compute_snorm_score.assert_awaited_once()


@pytest.mark.asyncio
async def test_verify_returns_mismatch_when_snorm_fails():
    user_id = uuid.uuid4()
    fake_vp = _make_fake_voiceprint(user_id)
    live_embedding = -np.asarray(fake_vp.embedding, dtype=np.float32)

    session = AsyncMock()
    session.add = MagicMock()
    service = VerificationService(session=session)

    user_repo_mock = AsyncMock()
    user_repo_mock.get_by_id.return_value = _make_fake_user(user_id)

    voiceprint_repo_mock = AsyncMock()
    voiceprint_repo_mock.get_by_user_id.return_value = fake_vp

    ver_log_repo_mock = AsyncMock()
    risk_log_repo_mock = AsyncMock()

    ecapa_mock = MagicMock()
    ecapa_mock.extract_embedding_with_confidence.return_value = (
        live_embedding,
        1.0,
        1.0,
    )

    snorm_mock = MagicMock()
    snorm_mock.compute_snorm_score = AsyncMock(return_value=(-1.0, 0.0, 0.2))
    snorm_mock.evaluate_decision.return_value = (DECISION_MISMATCH, False)

    with _mock_current_verification_pipeline(
        service,
        user_repo_mock,
        voiceprint_repo_mock,
        ver_log_repo_mock,
        risk_log_repo_mock,
        ecapa_mock,
        snorm_mock,
    ):
        result = await service.verify(
            user_id=user_id,
            audio_file=_make_upload_file(),
            layer1_score=0.05,
        )

    assert result.verified is False
    assert result.decision == DECISION_MISMATCH
    assert result.similarity_score < 0.0
    assert result.risk_level == "FRAUD_ALERT"


@pytest.mark.asyncio
async def test_verify_fraud_alert_when_layer1_score_high():
    """
    Even if speaker similarity is high enough, a high Layer 1 score
    should result in FRAUD_ALERT from the Risk Engine.
    """
    user_id = uuid.uuid4()
    fake_vp = _make_fake_voiceprint(user_id)
    live_embedding = np.asarray(fake_vp.embedding, dtype=np.float32)

    session = AsyncMock()
    session.add = MagicMock()
    service = VerificationService(session=session)

    user_repo_mock = AsyncMock()
    user_repo_mock.get_by_id.return_value = _make_fake_user(user_id)

    voiceprint_repo_mock = AsyncMock()
    voiceprint_repo_mock.get_by_user_id.return_value = fake_vp

    ver_log_repo_mock = AsyncMock()
    risk_log_repo_mock = AsyncMock()

    ecapa_mock = MagicMock()
    ecapa_mock.extract_embedding_with_confidence.return_value = (
        live_embedding,
        1.0,
        1.0,
    )

    snorm_mock = MagicMock()
    snorm_mock.compute_snorm_score = AsyncMock(return_value=(3.5, 0.0, 0.2))
    snorm_mock.evaluate_decision.return_value = (DECISION_VERIFIED, True)

    with _mock_current_verification_pipeline(
        service,
        user_repo_mock,
        voiceprint_repo_mock,
        ver_log_repo_mock,
        risk_log_repo_mock,
        ecapa_mock,
        snorm_mock,
    ):
        result = await service.verify(
            user_id=user_id,
            audio_file=_make_upload_file(),
            layer1_score=0.95,  # Layer 1 flags as AI-generated
        )

    assert result.risk_level == "FRAUD_ALERT"


@pytest.mark.asyncio
async def test_verify_logs_are_persisted_on_success():
    user_id = uuid.uuid4()
    fake_vp = _make_fake_voiceprint(user_id)
    live_embedding = np.asarray(fake_vp.embedding, dtype=np.float32)

    session = AsyncMock()
    session.add = MagicMock()
    service = VerificationService(session=session)

    user_repo_mock = AsyncMock()
    user_repo_mock.get_by_id.return_value = _make_fake_user(user_id)

    voiceprint_repo_mock = AsyncMock()
    voiceprint_repo_mock.get_by_user_id.return_value = fake_vp

    ver_log_repo_mock = AsyncMock()
    risk_log_repo_mock = AsyncMock()

    ecapa_mock = MagicMock()
    ecapa_mock.extract_embedding_with_confidence.return_value = (
        live_embedding,
        1.0,
        1.0,
    )

    snorm_mock = MagicMock()
    snorm_mock.compute_snorm_score = AsyncMock(return_value=(3.5, 0.0, 0.2))
    snorm_mock.evaluate_decision.return_value = (DECISION_VERIFIED, True)

    with _mock_current_verification_pipeline(
        service,
        user_repo_mock,
        voiceprint_repo_mock,
        ver_log_repo_mock,
        risk_log_repo_mock,
        ecapa_mock,
        snorm_mock,
    ):
        await service.verify(
            user_id=user_id,
            audio_file=_make_upload_file(),
            layer1_score=0.10,
        )

    ver_log_repo_mock.create.assert_called_once()
    risk_log_repo_mock.create.assert_called_once()
    assert session.commit.await_count == 2
    session.add.assert_called_once()


@pytest.mark.asyncio
async def test_high_tier_borderline_standard_pass_requires_step_up_otp():
    user_id = uuid.uuid4()
    fake_vp = _make_fake_voiceprint(user_id)
    live_embedding = np.asarray(fake_vp.embedding, dtype=np.float32)

    session = AsyncMock()
    session.add = MagicMock()
    service = VerificationService(session=session)

    user_repo_mock = AsyncMock()
    user_repo_mock.get_by_id.return_value = _make_fake_user(user_id)

    voiceprint_repo_mock = AsyncMock()
    voiceprint_repo_mock.get_by_user_id.return_value = fake_vp

    ver_log_repo_mock = AsyncMock()
    risk_log_repo_mock = AsyncMock()

    ecapa_mock = MagicMock()
    ecapa_mock.extract_embedding_with_confidence.return_value = (live_embedding, 1.0, 1.0)

    snorm_mock = MagicMock()
    snorm_mock.compute_snorm_score = AsyncMock(return_value=(3.5, 0.0, 0.2))
    snorm_mock.evaluate_decision.return_value = (DECISION_VERIFIED, True)

    with _mock_current_verification_pipeline(
        service,
        user_repo_mock,
        voiceprint_repo_mock,
        ver_log_repo_mock,
        risk_log_repo_mock,
        ecapa_mock,
        snorm_mock,
    ):
        result = await service.verify(
            user_id=user_id,
            audio_file=_make_upload_file(),
            transaction_type="fund_transfer",
            transaction_amount=25000,
            expected_phrase="transfer 25000 to account 1234",
            spoken_text="transfer twenty five thousand to account 1234",
        )

    assert result.transaction_tier == "HIGH"
    assert result.decision == DECISION_STEP_UP
    assert result.biometric_verified is True
    assert result.otp_required is True
    assert result.final_authorized is False


@pytest.mark.asyncio
async def test_low_tier_keeps_standard_pass_behavior():
    user_id = uuid.uuid4()
    fake_vp = _make_fake_voiceprint(user_id)
    live_embedding = np.asarray(fake_vp.embedding, dtype=np.float32)

    session = AsyncMock()
    session.add = MagicMock()
    service = VerificationService(session=session)

    user_repo_mock = AsyncMock()
    user_repo_mock.get_by_id.return_value = _make_fake_user(user_id)

    voiceprint_repo_mock = AsyncMock()
    voiceprint_repo_mock.get_by_user_id.return_value = fake_vp

    ver_log_repo_mock = AsyncMock()
    risk_log_repo_mock = AsyncMock()

    ecapa_mock = MagicMock()
    ecapa_mock.extract_embedding_with_confidence.return_value = (live_embedding, 1.0, 1.0)

    snorm_mock = MagicMock()
    snorm_mock.compute_snorm_score = AsyncMock(return_value=(3.5, 0.0, 0.2))
    snorm_mock.evaluate_decision.return_value = (DECISION_VERIFIED, True)

    with _mock_current_verification_pipeline(
        service,
        user_repo_mock,
        voiceprint_repo_mock,
        ver_log_repo_mock,
        risk_log_repo_mock,
        ecapa_mock,
        snorm_mock,
    ):
        result = await service.verify(
            user_id=user_id,
            audio_file=_make_upload_file(),
            transaction_type="balance_inquiry",
        )

    assert result.transaction_tier == "LOW"
    assert result.decision == DECISION_VERIFIED
    assert result.verified is True
    assert result.otp_required is False
    assert result.final_authorized is True


@pytest.mark.asyncio
async def test_transaction_phrase_mismatch_is_distinct_non_fraud_decision():
    user_id = uuid.uuid4()
    fake_vp = _make_fake_voiceprint(user_id)

    session = AsyncMock()
    session.add = MagicMock()
    service = VerificationService(session=session)

    user_repo_mock = AsyncMock()
    user_repo_mock.get_by_id.return_value = _make_fake_user(user_id)

    voiceprint_repo_mock = AsyncMock()
    voiceprint_repo_mock.get_by_user_id.return_value = fake_vp

    ver_log_repo_mock = AsyncMock()
    risk_log_repo_mock = AsyncMock()

    ecapa_mock = MagicMock()
    snorm_mock = MagicMock()
    snorm_mock.compute_snorm_score = AsyncMock()

    with _mock_current_verification_pipeline(
        service,
        user_repo_mock,
        voiceprint_repo_mock,
        ver_log_repo_mock,
        risk_log_repo_mock,
        ecapa_mock,
        snorm_mock,
    ):
        result = await service.verify(
            user_id=user_id,
            audio_file=_make_upload_file(),
            transaction_type="fund_transfer",
            transaction_amount=5000,
            expected_phrase="transfer 5000 to account 1234",
            spoken_text="check my balance",
        )

    assert result.transaction_tier == "MEDIUM"
    assert result.decision == DECISION_PHRASE_MISMATCH
    assert result.risk_level == "CLEAN"
    assert result.phrase_match is False
    assert result.biometric_verified is False
    snorm_mock.compute_snorm_score.assert_not_awaited()
    ecapa_mock.extract_embedding_with_confidence.assert_not_called()


@pytest.mark.asyncio
async def test_high_tier_strong_voice_match_still_requires_otp():
    user_id = uuid.uuid4()
    fake_vp = _make_fake_voiceprint(user_id)
    live_embedding = np.asarray(fake_vp.embedding, dtype=np.float32)

    session = AsyncMock()
    session.add = MagicMock()
    service = VerificationService(session=session)

    user_repo_mock = AsyncMock()
    user_repo_mock.get_by_id.return_value = _make_fake_user(user_id)

    voiceprint_repo_mock = AsyncMock()
    voiceprint_repo_mock.get_by_user_id.return_value = fake_vp

    ver_log_repo_mock = AsyncMock()
    risk_log_repo_mock = AsyncMock()

    ecapa_mock = MagicMock()
    ecapa_mock.extract_embedding_with_confidence.return_value = (live_embedding, 1.0, 1.0)

    snorm_mock = MagicMock()
    snorm_mock.compute_snorm_score = AsyncMock(return_value=(4.5, 0.0, 0.2))
    snorm_mock.evaluate_decision.return_value = (DECISION_VERIFIED, True)

    with _mock_current_verification_pipeline(
        service,
        user_repo_mock,
        voiceprint_repo_mock,
        ver_log_repo_mock,
        risk_log_repo_mock,
        ecapa_mock,
        snorm_mock,
    ):
        result = await service.verify(
            user_id=user_id,
            audio_file=_make_upload_file(),
            transaction_type="add_payee",
            expected_phrase="add payee account 9999",
            spoken_text="add payee account 9999",
        )

    assert result.transaction_tier == "HIGH"
    assert result.decision == DECISION_OTP_REQUIRED
    assert result.biometric_verified is True
    assert result.otp_required is True
    assert result.final_authorized is False
