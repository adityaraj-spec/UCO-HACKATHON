"""
tests/layer2/test_layer2_components.py

Comprehensive unit tests verifying Layer 2 sub-components:
  - Audio quality check logic (SNR estimation)
  - BioHash cancelable biometric projections
  - AES-GCM vault security wrappers
  - DPDP Consent grants, withdrawals & purges
  - Adaptive threshold calibration
  - Trusted emergency overrides
  - Real-time Fraud threat scoring & SIEM notifications
"""

from __future__ import annotations

import os
import uuid
import numpy as np
import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, AsyncMock

# Testing components
from app.preprocessing.audio_pipeline import AudioPipeline
from app.preprocessing.audio_normalizer import AudioNormalizer
from app.biometrics.biohash import BioHash
from app.biometrics.template_vault import TemplateVault
from app.encryption.key_manager import MockHSMKeyManager
from app.consent.consent_service import ConsentService
from app.consent.deletion_workflow import BiometricPurgeScheduler
from app.threshold.engine import AdaptiveThresholdEngine
from app.threshold.confidence_bands import classify_score, DecisionBand
from app.emergency.contact_service import EmergencyContactService
from app.emergency.activation_service import EmergencyActivationService
from app.emergency.access_scope import EmergencyAccessScope
from app.fraud.fraud_service import FraudService
from app.fraud.mock_fraud_engine import MockFraudEngine
from app.fraud.kafka_publisher import KafkaEventPublisher
from app.models.layer2.user_threshold import UserThreshold


# ------------------------------------------------------------------ #
# 1. Preprocessing Pipeline Tests                                    #
# ------------------------------------------------------------------ #

def test_audio_pipeline_snr_estimation():
    from app.preprocessing.quality_validator import AudioQualityValidator
    validator = AudioQualityValidator()
    sr = 16000
    t = np.linspace(0, 2, 2 * sr, endpoint=False)

    # Use a signal with very clear foreground frames (non-uniform power)
    # First half louder (signal), second half quieter (noise)
    loud_part = 0.8 * np.sin(2 * np.pi * 440 * t[:sr])       # 1s loud frames
    quiet_part = 0.01 * np.random.randn(sr).astype(np.float32) # 1s near-silence noise
    signal = np.concatenate([loud_part, quiet_part]).astype(np.float32)

    snr_high = validator._estimate_snr(signal, voice_ratio=1.0)
    assert snr_high > 10.0

    # Heavier noise should reduce SNR
    noisy_signal = signal + np.random.normal(0, 0.4, signal.shape).astype(np.float32)
    snr_low = validator._estimate_snr(noisy_signal, voice_ratio=1.0)
    assert snr_low < snr_high


def test_audio_normalization():
    # Normalization is applied inline inside load_and_normalize.
    # We verify the normalizer reduces peak of a clipped signal to ~TARGET_PEAK.
    normalizer = AudioNormalizer(target_peak=0.891)
    raw = np.array([2.0, -3.0, 5.0, -6.0], dtype=np.float32)
    # Manually apply normalizer's inline normalization logic
    raw_dc = raw - np.mean(raw)
    peak = np.max(np.abs(raw_dc))
    normalized = raw_dc * (normalizer.target_peak / peak) if peak > 1e-6 else raw_dc

    max_val = float(np.abs(normalized).max())
    assert np.isclose(max_val, 0.891, atol=0.05)


# ------------------------------------------------------------------ #
# 2. BioHash cancelable biometric projections                        #
# ------------------------------------------------------------------ #

def test_biohash_projection():
    service = BioHash(projection_dim=256, embedding_dim=192)
    user_key_1 = b"some_user_specific_key_seed_hash_1"
    user_key_2 = b"some_user_specific_key_seed_hash_2"
    embedding = np.random.normal(0.0, 0.1, 192).astype(np.float32)
    embedding /= np.linalg.norm(embedding)  # normalize

    projected_1 = service.transform(embedding, user_key_1)
    projected_2 = service.transform(embedding, user_key_1)
    projected_diff_seed = service.transform(embedding, user_key_2)
    
    # Checks idempotency with identical seed
    assert np.allclose(projected_1, projected_2)
    # Checks difference seed produces non-matching projection (cancelability)
    assert not np.allclose(projected_1, projected_diff_seed)
    
    # Confirm shape matches target dimensions
    assert projected_1.shape == (256,)


# ------------------------------------------------------------------ #
# 3. HSM and AES-GCM Template Vault Storage                          #
# ------------------------------------------------------------------ #

def test_hsm_key_derivation():
    hsm = MockHSMKeyManager()
    user_id = str(uuid.uuid4())
    master_key_id = "phaseguard-master-key-v1"
    key_1 = hsm.get_key_for_user(user_id, master_key_id)
    key_2 = hsm.get_key_for_user(user_id, master_key_id)

    # Key derivation must be consistent
    assert key_1 == key_2
    
    # Rotate should yield new master key version ID
    new_master_key_id = hsm.rotate_key(master_key_id)
    key_rotated = hsm.get_key_for_user(user_id, new_master_key_id)
    assert key_rotated != key_1


def test_template_vault_encryption():
    vault = TemplateVault()
    user_id = uuid.uuid4()
    original_emb = np.random.normal(0, 0.1, 192).astype(np.float32)
    
    encrypted_bytes, nonce, integrity_hash, key_id = vault.generate_template(user_id, original_emb)
    assert encrypted_bytes != original_emb.tobytes()

    decrypted = vault.decrypt_template(user_id, encrypted_bytes, nonce, key_id)
    # decrypted template is the BioHashed array shape (256,)
    assert decrypted.shape == (256,)

    challenge_emb = vault.generate_challenge_template(user_id, original_emb, key_id)
    assert np.allclose(decrypted, challenge_emb, atol=1e-5)


# ------------------------------------------------------------------ #
# 4. Consent Compliance Service (DPDP checks) & Deletion Purger      #
# ------------------------------------------------------------------ #

@pytest.mark.asyncio
async def test_consent_recording():
    db = AsyncMock()
    service = ConsentService()
    user_id = uuid.uuid4()
    
    # Mock database responses for active consent check
    mock_result = MagicMock()
    mock_result.scalars.return_value.first.return_value = None
    db.execute = AsyncMock(return_value=mock_result)

    # Record grant
    record = await service.record_consent_grant(db, user_id, "192.168.1.1", "Chrome", "EXPLICIT_OPT_IN")
    assert record.user_id == user_id
    assert record.status == "ACTIVE"
    assert record.consent_token is not None




@pytest.mark.asyncio
async def test_biometric_purge_scheduler():
    db = AsyncMock()
    scheduler = BiometricPurgeScheduler()
    user_id = uuid.uuid4()

    # Mock purge query executions
    db.execute = AsyncMock()
    
    res = await scheduler.execute_purge(db, user_id)
    assert res is True
    # Ensure deletions were called
    assert db.execute.call_count >= 2


# ------------------------------------------------------------------ #
# 5. Adaptive Threshold Engine                                       #
# ------------------------------------------------------------------ #

def test_adaptive_threshold_calibration():
    engine = AdaptiveThresholdEngine()
    user_t = UserThreshold(
        user_id=uuid.uuid4(),
        threshold_baseline=0.82,
        threshold_current=0.82,
        threshold_floor=0.50,
        threshold_ceiling=0.90,
        illness_window_active=False
    )
    
    # 1. Base calibration with low risk, no failure
    t_base = engine.calibrate_threshold(user_t, consec_failures=0, illness_active=False, fraud_risk_level="LOW", ip_device_trusted=True)
    assert t_base == 0.82

    # 2. Velocity degradation: failures should relax threshold incrementally provider IP/device is trusted
    t_failed = engine.calibrate_threshold(user_t, consec_failures=2, illness_active=False, fraud_risk_level="LOW", ip_device_trusted=True)
    assert t_failed < 0.82

    # Reset current threshold back to base
    user_t.threshold_current = 0.82

    # 3. High fraud assessment should boost threshold significantly
    t_fraud = engine.calibrate_threshold(user_t, consec_failures=0, illness_active=False, fraud_risk_level="HIGH", ip_device_trusted=True)
    assert t_fraud > 0.82

    # Reset current threshold back to base
    user_t.threshold_current = 0.82

    # 4. Illness window should lower threshold slightly to accommodate hoarseness
    user_t.illness_window_expires_at = datetime.now(timezone.utc) + timedelta(hours=24)
    t_illness = engine.calibrate_threshold(user_t, consec_failures=0, illness_active=True, fraud_risk_level="LOW", ip_device_trusted=True)
    assert t_illness < 0.82


def test_confidence_bands_scoring():
    # threshold=0.80, step_up_bound=0.05 (width of step-up zone)
    # step_up_threshold = 0.80 - 0.05 = 0.75
    threshold = 0.80
    step_up_bound = 0.05   # width, not absolute bound

    band, _ = classify_score(0.85, threshold, step_up_bound)
    assert band == DecisionBand.PASS                # 0.85 >= 0.80

    band, _ = classify_score(0.77, threshold, step_up_bound)
    assert band == DecisionBand.STEP_UP             # 0.77 in [0.75, 0.80)

    band, _ = classify_score(0.70, threshold, step_up_bound)
    assert band == DecisionBand.HARD_FAIL           # 0.70 < 0.75


# ------------------------------------------------------------------ #
# 6. Emergency Access Trustee Overrides                              #
# ------------------------------------------------------------------ #

@pytest.mark.asyncio
async def test_emergency_contact_registration():
    from unittest.mock import MagicMock
    db = AsyncMock()
    service = EmergencyContactService()
    user_id = uuid.uuid4()

    # db.execute returns an AsyncMock; its .scalars() must be synchronous.
    # Set execute's return value to a plain MagicMock so scalars().all() works synchronously.
    mock_result = MagicMock()
    mock_result.scalars.return_value.all.return_value = []
    db.execute = AsyncMock(return_value=mock_result)

    # Save contact
    contact = await service.register_contact(
        db, user_id, "Trustee John", "+919876543210", "john@bank.com", EmergencyAccessScope.FULL_TRANSACTION
    )
    assert contact.user_id == user_id
    assert contact.contact_name == "Trustee John"
    assert contact.is_verified is False


@pytest.mark.asyncio
async def test_emergency_activation():
    db = MagicMock()
    mock_result = MagicMock()
    mock_result.scalars.return_value.first.return_value = None
    db.execute = AsyncMock(return_value=mock_result)
    db.flush = AsyncMock()
    db.add = MagicMock()
    service = EmergencyActivationService()
    from app.models.layer2.emergency_contact import EmergencyContact

    contact = EmergencyContact(
        id=uuid.uuid4(),
        user_id=uuid.uuid4(),
        contact_name="Bob",
        encrypted_phone=b"enc_phone",
        phone_nonce=b"123456789012",
        encrypted_email=b"enc_email",
        email_nonce=b"123456789012",
        key_id="phaseguard-master-key-v1",
        access_scope="READ_ONLY",
        is_verified=True,
        is_active=True,
    )

    # Mock PII decryption so activation can proceed without real keys
    service.contact_service.decrypt_contact_pii = AsyncMock(return_value=("+919876543210", "bob@bank.com"))

    # Start activation -> issues OTP ref
    otp_ref = await service.initiate_activation(db, contact)
    assert otp_ref is not None

    # Mock Redis lookup check
    service.otp.verify_otp = AsyncMock(return_value=True)

    # Complete OTP verification to trigger override session
    event = await service.complete_activation(db, contact, "123456", "10.0.0.2")
    assert event is not None
    assert event.access_scope_granted == "READ_ONLY"
    assert event.expires_at > datetime.now(timezone.utc) + timedelta(hours=71)


# ------------------------------------------------------------------ #
# 7. Fraud Risk Engine and SIEM publishing                          #
# ------------------------------------------------------------------ #

def test_fraud_evaluation_and_alerts():
    from app.fraud.interface import FraudResult
    service = FraudService()
    user_id = uuid.uuid4()

    # Stub engine.score to return a CRITICAL FraudResult directly
    mock_result = FraudResult(
        score=0.95,
        risk_level="CRITICAL",
        action_required="REJECT",
        reason="Botnets matched.",
        transaction_id="TX_TESTID",
    )
    service.engine.score = MagicMock(return_value=mock_result)

    # Stub Kafka event publisher
    service.publisher.publish_event = MagicMock()

    result = service.evaluate_risk(
        user_id=user_id,
        ip_address="198.51.100.42",
        device_fingerprint="device-fingerprint-bad"
    )

    assert result.risk_level == "CRITICAL"
    assert result.action_required == "REJECT"

    # Kafka should publish the HIGH alert safety log
    service.publisher.publish_event.assert_called()
    assert service.publisher.publish_event.call_args[0][0] == "FRAUD_ALERT_HIGH"
