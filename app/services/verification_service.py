"""
app/services/verification_service.py

Business logic for the production-grade Layer 2 speaker verification pipeline:
- Audio quality & VAD duration check
- Perceptual audio hashing replay detection
- Challenge phrase verification (ASR)
- BioHash generation & Hamming distance similarity mapping
- Adaptive Score Normalisation (s-norm)
- Circuit-breaker wrapped FAISS 1-to-N search
- Immutable compliance audit logging
"""

import uuid
import hashlib
import time
import numpy as np
from fastapi import UploadFile, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.logging import get_logger
from app.ml.ecapa_service import ECAPAService, get_ecapa_service
from app.models.verification_audit_log import VerificationAuditLog
from app.repositories.risk_repository import RiskLogRepository
from app.repositories.user_repository import UserRepository
from app.repositories.verification_repository import VerificationLogRepository
from app.repositories.voiceprint_repository import VoiceprintRepository
from app.schemas.verification import VerificationResult
from app.services.biohash_service import BioHashService
from app.services.circuit_breaker import FAISSCircuitBreaker
from app.services.redis_service import redis_service
from app.services.risk_engine import compute_risk
from app.services.snorm_service import SNormService
from app.services.transaction_policy import (
    DECISION_PHRASE_MISMATCH,
    apply_transaction_policy,
    parse_transaction_context,
    resolve_transaction_tier,
    tier_requires_phrase,
)
from app.utils.asr_verifier import verify_spoken_challenge, verify_transaction_phrase
from app.utils.audio_quality import (
    diarise_and_extract_dominant_speaker,
    compute_perceptual_audio_hash,
)
from app.utils.audio_utils import cleanup_temp_file, save_upload_to_temp, load_waveform
from app.utils.exceptions import UserNotFoundError, VoiceprintNotFoundError

settings = get_settings()
log = get_logger(__name__)

DECISION_VERIFIED = "verified"
DECISION_MISMATCH = "mismatch"
DECISION_STEP_UP = "step_up"
DECISION_REPLAY = "REPLAY_DETECTED"
DECISION_LAYER1_REJECTED = "layer1_rejected"


class VerificationService:
    """Orchestrates production-grade Layer 2 speaker verification."""

    def __init__(
        self,
        session: AsyncSession,
        ecapa_service: ECAPAService | None = None,
        biohash_service: BioHashService | None = None,
        snorm_service: SNormService | None = None,
    ) -> None:
        self.session = session
        self.user_repo = UserRepository(session)
        self.voiceprint_repo = VoiceprintRepository(session)
        self.verification_log_repo = VerificationLogRepository(session)
        self.risk_log_repo = RiskLogRepository(session)
        self.ecapa_service = ecapa_service or get_ecapa_service()
        self.biohash_service = biohash_service or BioHashService()
        self.snorm_service = snorm_service or SNormService(self.biohash_service)
        self.circuit_breaker = FAISSCircuitBreaker()

    async def verify(
        self,
        user_id: uuid.UUID,
        audio_file: UploadFile,
        layer1_score: float = 0.0,
        challenge_phrase: str | None = None,
        expected_phrase: str | None = None,
        spoken_text: str | None = None,
        transaction_type: str | None = None,
        transaction_amount: float | str | None = None,
    ) -> VerificationResult:
        """
        Production-grade verification pipeline with BioHash, s-norm, audio quality gating,
        replay detection, and immutable audit logging.
        """
        total_started = time.perf_counter()
        stage_timings: dict[str, float] = {}

        def record_stage(name: str, started: float) -> None:
            stage_timings[name] = round((time.perf_counter() - started) * 1000.0, 2)

        stage_started = time.perf_counter()
        user = await self.user_repo.get_by_id(user_id)
        if user is None:
            raise UserNotFoundError(str(user_id))

        voiceprint = await self.voiceprint_repo.get_by_user_id(user_id)
        if voiceprint is None:
            raise VoiceprintNotFoundError(str(user_id))
        record_stage("lookup", stage_started)

        transaction_context = parse_transaction_context(
            transaction_type, transaction_amount, expected_phrase
        )
        transaction_tier = resolve_transaction_tier(transaction_context)
        response_transaction_amount = (
            float(transaction_context.transaction_amount)
            if transaction_context.transaction_amount is not None
            else None
        )

        if layer1_score > settings.LAYER1_FRAUD_THRESHOLD:
            risk_result = compute_risk(layer1_score=layer1_score, speaker_similarity=0.0)
            await self.verification_log_repo.create(
                user_id=user_id,
                similarity_score=0.0,
                decision=DECISION_LAYER1_REJECTED,
            )
            await self.risk_log_repo.create(
                user_id=user_id,
                risk_score=risk_result.risk_score,
                risk_level=risk_result.risk_level,
            )
            await self.session.commit()
            await self._record_audit_log(
                user_id,
                DECISION_LAYER1_REJECTED,
                0.0,
                None,
                voiceprint.model_version,
                "layer1",
                transaction_context=transaction_context,
                transaction_tier=transaction_tier,
                phrase_match=None,
                biometric_decision=DECISION_LAYER1_REJECTED,
            )
            stage_timings["total"] = round((time.perf_counter() - total_started) * 1000.0, 2)
            return VerificationResult(
                user_id=user_id,
                similarity_score=0.0,
                verified=False,
                decision=DECISION_LAYER1_REJECTED,
                layer1_score=layer1_score,
                risk_score=risk_result.risk_score,
                risk_level=risk_result.risk_level,
                stage_timings_ms=stage_timings,
                transaction_type=transaction_context.transaction_type,
                transaction_amount=response_transaction_amount,
                transaction_tier=transaction_tier,
                phrase_match=None,
                biometric_verified=False,
                otp_required=False,
                final_authorized=False,
            )

        temp_path: str | None = None
        try:
            stage_started = time.perf_counter()
            temp_path = await save_upload_to_temp(audio_file)
            waveform, sr = load_waveform(temp_path)
            record_stage("audio_io", stage_started)

            # 1. Neural Speaker Diarisation & Target Speaker Isolation (2s min check on target speech)
            stage_started = time.perf_counter()
            dominant_waveform, duration = diarise_and_extract_dominant_speaker(
                waveform, sr, enrolled_embedding=voiceprint.embedding
            )
            record_stage("diarisation", stage_started)

            # 2. Challenge-Response Phrase Verification
            stage_started = time.perf_counter()
            if challenge_phrase and settings.ENABLE_CHALLENGE_ASR:
                if not verify_spoken_challenge(spoken_text, challenge_phrase):
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Challenge text mismatch: Spoken phrase does not match expected challenge.",
                    )
            record_stage("challenge", stage_started)

            # 3. Audio Perceptual Replay Detection
            stage_started = time.perf_counter()
            audio_hash = compute_perceptual_audio_hash(dominant_waveform, sr)
            if settings.ENABLE_PERCEPTUAL_REPLAY_CHECK:
                if not redis_service.check_and_store_audio_hash(str(user_id), audio_hash, ttl_sec=settings.AUDIO_HASH_TTL_SEC):
                    record_stage("replay_hash", stage_started)
                    log.warning(f"Replay attack detected for user_id={user_id}")
                    # Audit log replay attack
                    audit_started = time.perf_counter()
                    await self._record_audit_log(
                        user_id,
                        DECISION_REPLAY,
                        0.0,
                        0.0,
                        voiceprint.model_version,
                        temp_path,
                        transaction_context=transaction_context,
                        transaction_tier=transaction_tier,
                        phrase_match=None,
                        biometric_decision=DECISION_REPLAY,
                    )
                    record_stage("audit", audit_started)
                    stage_timings["total"] = round((time.perf_counter() - total_started) * 1000.0, 2)
                    return VerificationResult(
                        user_id=user_id,
                        similarity_score=0.0,
                        verified=False,
                        decision=DECISION_REPLAY,
                        layer1_score=layer1_score,
                        risk_score=100.0,
                        risk_level="FRAUD_ALERT",
                        stage_timings_ms=stage_timings,
                        transaction_type=transaction_context.transaction_type,
                        transaction_amount=response_transaction_amount,
                        transaction_tier=transaction_tier,
                        phrase_match=None,
                        biometric_verified=False,
                        otp_required=False,
                        final_authorized=False,
                    )
            record_stage("replay_hash", stage_started)

            # 4. Dynamic transaction phrase verification, when required.
            stage_started = time.perf_counter()
            phrase_match: bool | None = None
            if tier_requires_phrase(transaction_tier):
                phrase_match = verify_transaction_phrase(
                    spoken_text, transaction_context.expected_phrase or ""
                )
                if not phrase_match:
                    risk_result = compute_risk(layer1_score=layer1_score, speaker_similarity=1.0)
                    await self.verification_log_repo.create(
                        user_id=user_id,
                        similarity_score=0.0,
                        decision=DECISION_PHRASE_MISMATCH,
                    )
                    await self.risk_log_repo.create(
                        user_id=user_id,
                        risk_score=risk_result.risk_score,
                        risk_level=risk_result.risk_level,
                    )
                    await self.session.commit()
                    await self._record_audit_log(
                        user_id,
                        DECISION_PHRASE_MISMATCH,
                        0.0,
                        None,
                        voiceprint.model_version,
                        temp_path,
                        transaction_context=transaction_context,
                        transaction_tier=transaction_tier,
                        phrase_match=False,
                        biometric_decision=None,
                    )
                    record_stage("transaction_phrase", stage_started)
                    stage_timings["total"] = round((time.perf_counter() - total_started) * 1000.0, 2)
                    return VerificationResult(
                        user_id=user_id,
                        similarity_score=0.0,
                        verified=False,
                        decision=DECISION_PHRASE_MISMATCH,
                        layer1_score=layer1_score,
                        risk_score=risk_result.risk_score,
                        risk_level=risk_result.risk_level,
                        stage_timings_ms=stage_timings,
                        transaction_type=transaction_context.transaction_type,
                        transaction_amount=response_transaction_amount,
                        transaction_tier=transaction_tier,
                        phrase_match=False,
                        biometric_verified=False,
                        otp_required=False,
                        final_authorized=False,
                    )
            record_stage("transaction_phrase", stage_started)

            # 5. Extract Embedding & Model-Derived Confidence Score
            stage_started = time.perf_counter()
            live_emb, model_confidence, raw_norm = self.ecapa_service.extract_embedding_with_confidence(temp_path)
            record_stage("embedding_extraction", stage_started)

            stage_started = time.perf_counter()
            live_biohash = self.biohash_service.compute_biohash(live_emb)

            # Retrieve enrolled BioHash (or compute from stored embedding if missing)
            enrolled_biohash = voiceprint.biohash
            if not enrolled_biohash:
                enrolled_biohash = self.biohash_service.compute_biohash(voiceprint.embedding)
                voiceprint.biohash = enrolled_biohash

            # 6. Compute BioHash Cosine Equivalence Similarity
            raw_similarity = self.biohash_service.biohash_similarity(live_biohash, enrolled_biohash)
            record_stage("biohash_similarity", stage_started)

            # 7. Adaptive Score Normalisation (s-norm)
            stage_started = time.perf_counter()
            snorm_score, _, _ = await self.snorm_service.compute_snorm_score(
                self.session, raw_similarity, live_biohash
            )
            decision, verified = self.snorm_service.evaluate_decision(snorm_score)
            record_stage("snorm", stage_started)

            # 7. Model-Derived Embedding Confidence Auto-Escalation
            if model_confidence < settings.EMBEDDING_CONFIDENCE_THRESHOLD and decision == DECISION_VERIFIED:
                log.info(f"Low model embedding confidence ({model_confidence:.2f} < {settings.EMBEDDING_CONFIDENCE_THRESHOLD}); auto-escalating decision to STEP_UP")
                decision = DECISION_STEP_UP
                verified = False

            transaction_decision = apply_transaction_policy(
                base_decision=decision,
                base_verified=verified,
                snorm_score=snorm_score,
                tier=transaction_tier,
                phrase_match=phrase_match,
            )
            decision = transaction_decision.decision
            verified = transaction_decision.verified

            # Compute combined risk assessment
            stage_started = time.perf_counter()
            risk_result = compute_risk(layer1_score=layer1_score, speaker_similarity=raw_similarity)

            # Persist Logs
            await self.verification_log_repo.create(
                user_id=user_id,
                similarity_score=raw_similarity,
                decision=decision,
            )
            await self.risk_log_repo.create(
                user_id=user_id,
                risk_score=risk_result.risk_score,
                risk_level=risk_result.risk_level,
            )
            await self.session.commit()
            record_stage("risk_and_logs", stage_started)

            # Record Immutable Audit Log
            stage_started = time.perf_counter()
            await self._record_audit_log(
                user_id,
                decision,
                raw_similarity,
                snorm_score,
                voiceprint.model_version,
                temp_path,
                transaction_context=transaction_context,
                transaction_tier=transaction_tier,
                phrase_match=transaction_decision.phrase_match,
                biometric_decision=DECISION_VERIFIED if transaction_decision.biometric_verified else decision,
            )
            record_stage("audit", stage_started)
            stage_timings["total"] = round((time.perf_counter() - total_started) * 1000.0, 2)

            log.info(
                f"Verification complete for user_id={user_id}: "
                f"raw_sim={raw_similarity:.4f}, snorm={snorm_score:.2f}, "
                f"tier={transaction_tier}, decision={decision}, "
                f"otp_required={transaction_decision.otp_required}"
            )

            return VerificationResult(
                user_id=user_id,
                similarity_score=raw_similarity,
                verified=verified,
                decision=decision,
                layer1_score=layer1_score,
                risk_score=risk_result.risk_score,
                risk_level=risk_result.risk_level,
                stage_timings_ms=stage_timings,
                transaction_type=transaction_context.transaction_type,
                transaction_amount=response_transaction_amount,
                transaction_tier=transaction_tier,
                phrase_match=transaction_decision.phrase_match,
                biometric_verified=transaction_decision.biometric_verified,
                otp_required=transaction_decision.otp_required,
                final_authorized=transaction_decision.final_authorized,
            )
        except (UserNotFoundError, VoiceprintNotFoundError, HTTPException):
            await self.session.rollback()
            raise
        except Exception:
            await self.session.rollback()
            log.exception(f"Verification failed for user_id={user_id}")
            raise
        finally:
            cleanup_temp_file(temp_path)

    async def _record_audit_log(
        self,
        user_id: uuid.UUID,
        decision: str,
        raw_sim: float,
        snorm: float,
        model_version: str,
        audio_path: str,
        transaction_context=None,
        transaction_tier: str | None = None,
        phrase_match: bool | None = None,
        biometric_decision: str | None = None,
    ):
        """Insert append-only immutable audit log."""
        try:
            audit_material = (
                f"{user_id}:{audio_path}:{raw_sim}:{transaction_context}:"
                f"{transaction_tier}:{phrase_match}:{biometric_decision}"
            )
            req_hash = hashlib.sha256(audit_material.encode()).hexdigest()[:64]
            audit_entry = VerificationAuditLog(
                user_id=user_id,
                decision=decision,
                raw_similarity=raw_sim,
                snorm_score=snorm,
                model_version=model_version,
                request_id_hash=req_hash,
            )
            self.session.add(audit_entry)
            await self.session.commit()
            if transaction_context and transaction_context.has_transaction:
                log.info(
                    "Verification audit metadata user_id=%s tier=%s transaction_type=%s "
                    "amount=%s phrase_match=%s biometric_decision=%s final_decision=%s",
                    user_id,
                    transaction_tier,
                    transaction_context.transaction_type,
                    transaction_context.transaction_amount,
                    phrase_match,
                    biometric_decision,
                    decision,
                )
        except Exception as e:
            log.error(f"Failed to record audit log: {e}")

    async def identify(
        self,
        audio_file: UploadFile,
        k: int = 5,
    ) -> list[dict]:
        """
        Speaker Identification using Circuit-Breaker wrapped FAISS search.
        Falls back to PostgreSQL query if Circuit Breaker is OPEN.
        """
        temp_path: str | None = None
        try:
            temp_path = await save_upload_to_temp(audio_file)
            live_emb = self.ecapa_service.extract_embedding(temp_path)

            if self.circuit_breaker.can_execute():
                try:
                    from app.services.faiss_service import faiss_service
                    results = faiss_service.search(live_emb.tolist(), k=k)
                    self.circuit_breaker.record_success()
                    return [{"user_id": uid, "similarity_score": score} for uid, score in results]
                except Exception as exc:
                    log.error(f"FAISS search failed: {exc}")
                    self.circuit_breaker.record_failure()

            # Circuit breaker fallback to PostgreSQL
            log.info("Circuit Breaker fallback: querying PostgreSQL directly")
            db_voiceprints = await self.voiceprint_repo.get_all()
            scored = []
            for vp in db_voiceprints:
                sim = self.ecapa_service.compute_cosine_similarity(live_emb, np.asarray(vp.embedding))
                scored.append({"user_id": vp.user_id, "similarity_score": sim})
            scored.sort(key=lambda x: x["similarity_score"], reverse=True)
            return scored[:k]
        finally:
            cleanup_temp_file(temp_path)
