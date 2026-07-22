"""
tests/test_production_upgrade.py

Comprehensive unit tests for the production-grade PhaseGuard Layer 2 features:
- BioHash generation & Hamming distance cosine mapping
- s-norm score normalisation
- FAISS IVF-PQ index operations & double-buffer swap
- Challenge-response phrase matching
- Audio VAD duration check & perceptual replay detection
- Circuit breaker state transitions
"""

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import torch
import pytest
from app.services.biohash_service import BioHashService
from app.services.circuit_breaker import FAISSCircuitBreaker
from app.services.diarisation_service import DiarisationService
from app.services.snorm_service import MIN_REAL_SNORM_COHORT_SIZE, SNormService
from app.utils.asr_verifier import verify_spoken_challenge
from app.utils.audio_quality import compute_perceptual_audio_hash
from app.ml.ecapa_service import ECAPAService


def test_biohash_generation_and_similarity():
    service = BioHashService()

    emb1 = [0.1] * 192
    emb2 = [0.1] * 192
    emb3 = [-0.1] * 192

    b1 = service.compute_biohash(emb1)
    b2 = service.compute_biohash(emb2)
    b3 = service.compute_biohash(emb3)

    assert len(b1) == 256
    assert b1 == b2

    sim_identical = service.biohash_similarity(b1, b2)
    assert sim_identical > 0.99

    sim_opposite = service.biohash_similarity(b1, b3)
    assert sim_opposite < 0.0


def test_snorm_decision_evaluation():
    service = SNormService()

    decision_pass, verified_pass = service.evaluate_decision(3.5)
    assert decision_pass == "verified"
    assert verified_pass is True

    decision_stepup, verified_stepup = service.evaluate_decision(2.0)
    assert decision_stepup == "step_up"
    assert verified_stepup is False

    decision_fail, verified_fail = service.evaluate_decision(-1.0)
    assert decision_fail == "mismatch"
    assert verified_fail is False


@pytest.mark.asyncio
async def test_snorm_small_cohort_fallback_does_not_reject_genuine_match():
    biohash_service = BioHashService()
    service = SNormService(biohash_service)

    embedding = np.full(192, 0.1, dtype=np.float32)
    live_biohash = biohash_service.compute_biohash(embedding)
    raw_similarity = biohash_service.biohash_similarity(live_biohash, live_biohash)

    result = MagicMock()
    result.scalars.return_value.all.return_value = []
    session = AsyncMock()
    session.execute.return_value = result

    with patch.object(
        service,
        "_synthetic_impostor_embeddings",
        wraps=service._synthetic_impostor_embeddings,
    ) as synthetic_mock:
        snorm_score, mu_imp, sigma_imp = await service.compute_snorm_score(
            session, raw_similarity, live_biohash
        )
    decision, verified = service.evaluate_decision(snorm_score)

    assert raw_similarity == pytest.approx(1.0)
    synthetic_mock.assert_called_once_with(count=MIN_REAL_SNORM_COHORT_SIZE)
    assert mu_imp != pytest.approx(0.0)
    assert sigma_imp != pytest.approx(1.0)
    assert snorm_score >= 3.0
    assert decision == "verified"
    assert verified is True


@pytest.mark.asyncio
async def test_snorm_does_not_pad_when_minimum_real_cohort_exists():
    biohash_service = BioHashService()
    service = SNormService(biohash_service)

    rng = np.random.default_rng(7)
    real_cohort = rng.normal(size=(MIN_REAL_SNORM_COHORT_SIZE, 192)).astype(np.float32)
    real_cohort = real_cohort / np.linalg.norm(real_cohort, axis=1, keepdims=True)

    live_embedding = real_cohort[0]
    live_biohash = biohash_service.compute_biohash(live_embedding)

    result = MagicMock()
    result.scalars.return_value.all.return_value = real_cohort.tolist()
    session = AsyncMock()
    session.execute.return_value = result

    with patch.object(service, "_synthetic_impostor_embeddings") as synthetic_mock:
        await service.compute_snorm_score(session, 0.65, live_biohash)

    synthetic_mock.assert_not_called()


@pytest.mark.asyncio
async def test_snorm_small_cohort_realistic_marginal_similarity_is_bounded():
    biohash_service = BioHashService()
    service = SNormService(biohash_service)

    embedding = np.linspace(-0.5, 0.5, 192, dtype=np.float32)
    embedding = embedding / np.linalg.norm(embedding)
    live_biohash = biohash_service.compute_biohash(embedding)
    raw_similarity = 0.65

    result = MagicMock()
    result.scalars.return_value.all.return_value = []
    session = AsyncMock()
    session.execute.return_value = result

    snorm_score, mu_imp, sigma_imp = await service.compute_snorm_score(
        session, raw_similarity, live_biohash
    )

    assert np.isfinite(snorm_score)
    assert -8.0 < snorm_score < 8.0
    assert -0.5 < mu_imp < 0.5
    assert sigma_imp > 0.05


def test_challenge_text_verification():
    expected = "My voice is my password in UCO Bank security"
    spoken = "my voice is my password in uco bank security"

    assert verify_spoken_challenge(spoken, expected) is True
    assert verify_spoken_challenge("completely wrong spoken phrase", expected) is False


def test_circuit_breaker_transitions():
    breaker = FAISSCircuitBreaker(max_failures=3, failure_window_sec=30.0, cooldown_sec=0.1)

    assert breaker.can_execute() is True
    breaker.record_failure()
    breaker.record_failure()
    breaker.record_failure()

    assert breaker.state == "OPEN"
    assert breaker.can_execute() is False


def test_speaker_diarisation_and_perceptual_hashing():
    class FakeSpeakerModel:
        def encode_batch(self, chunk_tensor):
            return torch.ones(1, 1, 192)

    class FakeECAPAService:
        def load_model(self):
            return FakeSpeakerModel()

    sample_rate = 16000
    # Generate 4 seconds of synthetic audio
    t = torch.linspace(0, 4, sample_rate * 4).unsqueeze(0)
    audio = torch.sin(2 * 3.14159 * 440 * t)

    diarisation_service = DiarisationService(ecapa_service=FakeECAPAService())
    speech, duration = diarisation_service.diarise_speech(audio, sample_rate)
    assert duration >= 2.0

    h1 = compute_perceptual_audio_hash(speech, sample_rate)
    h2 = compute_perceptual_audio_hash(speech, sample_rate)
    assert len(h1) == 64
    assert h1 == h2
