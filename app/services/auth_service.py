"""
app/services/auth_service.py

Audio speaker verification and multi-factor authentication (Layer 2 orchestration).

Integrates:
  - JTI Replay Guard & challenge token validation
  - Fraud risk & device profiling
  - Audio preprocessing pipeline
  - WebRTC/AASIST liveness & anti-spoof checks
  - BiosHash projection + AES-GCM decryptions
  - Ensemble weighted similarity scoring (anchor + rolling pool)
  - Adaptive per-user thresholds & confidence bands
  - Rolling pool FIFO cache rotation (safe drift bounds)
  - Immutable audit logs
"""

from __future__ import annotations

import logging
import os
import uuid
import numpy as np
from datetime import datetime, timezone
from fastapi import UploadFile

from app.antispoof.challenge_service import get_challenge_service
from app.antispoof.liveness_verifier import get_antispoof_engine
from app.audit.event_types import AuditEventType
from app.audit.vault import get_audit_vault
from app.biometrics.template_vault import get_template_vault
from app.cache.session_cache import SessionCache
from app.core.config import get_settings
from app.database.session import AsyncSession
from app.embeddings.rolling_manager import get_rolling_manager
from app.embeddings.weighted_scorer import get_weighted_scorer
from app.fraud.fraud_service import get_fraud_service
from app.ml.ecapa_service import get_ecapa_service
from app.preprocessing.audio_pipeline import get_audio_pipeline
from app.repositories.layer2.anchor_embedding_repo import AnchorEmbeddingRepository
from app.repositories.layer2.rolling_embedding_repo import RollingEmbeddingRepository
from app.repositories.layer2.threshold_repo import UserThresholdRepository
from app.repositories.layer2.auth_history_repo import AuthHistoryRepository
from app.threshold.engine import get_threshold_engine
from app.threshold.confidence_bands import classify_score, DecisionBand
from app.utils.audio_utils import cleanup_temp_file, save_upload_to_temp

logger = logging.getLogger(__name__)
settings = get_settings()


class AuthService:
    """Core voice biometric authentication service."""

    def __init__(self, session: AsyncSession) -> None:
        self.db = session
        self.anchor_repo = AnchorEmbeddingRepository(session)
        self.rolling_repo = RollingEmbeddingRepository(session)
        self.threshold_repo = UserThresholdRepository(session)
        self.history_repo = AuthHistoryRepository(session)
        self.challenge_service = get_challenge_service()
        self.session_cache = SessionCache()
        self.fraud_service = get_fraud_service()
        self.audio_pipeline = get_audio_pipeline()
        self.antispoof_engine = get_antispoof_engine()
        self.ecapa = get_ecapa_service()
        self.vault = get_template_vault()
        self.scorer = get_weighted_scorer()
        self.threshold_engine = get_threshold_engine()
        self.rolling_manager = get_rolling_manager()
        self.audit = get_audit_vault()

    async def authenticate(
        self,
        user_id: uuid.UUID,
        session_id: str,
        challenge_token: str,
        audio_file: UploadFile,
        ip_address: str,
        device_fingerprint: str,
        latitude: float | None = None,
        longitude: float | None = None,
    ) -> dict:
        """
        Authenticate a live voice sample against the user ensemble.
        """
        now_utc = datetime.now(timezone.utc)
        logger.info("Initializing voice auth transaction for user ID: %s", user_id)

        # ------------------------------------------------------------------ #
        # Step 1: Challenge Replay Guard (validate signing + consume JTI)    #
        # ------------------------------------------------------------------ #
        # Extract JWT claims first for replay JTI checking
        import jwt
        try:
            claims = jwt.decode(challenge_token, settings.JWT_SECRET_KEY, algorithms=["HS256"])
            jti = claims.get("jti")
            if not jti:
                raise ValueError("Missing challenge JTI signature.")
        except jwt.PyJWTError as e:
            raise ValueError(f"Challenge signature error: {e}")

        # Consume challenge token in Redis
        is_fresh = await self.session_cache.verify_and_consume_jti(jti)
        if not is_fresh:
            await self._log_auth_failure(
                user_id=user_id, session_id=session_id, phrase_id=claims.get("phrase_id", "UNKNOWN"),
                reason="Challenge token reuse/replay detected.", ip=ip_address
            )
            return {"authenticated": False, "decision": "REJECT", "reason": "Liveness verification challenge token already consumed."}

        # Verify cryptographic binding constraints
        is_token_val, phrase_id, reason_fail = self.challenge_service.verify_challenge_token(
            challenge_token, user_id, session_id
        )
        if not is_token_val:
            await self._log_auth_failure(
                user_id=user_id, session_id=session_id, phrase_id="UNKNOWN",
                reason=reason_fail, ip=ip_address
            )
            return {"authenticated": False, "decision": "REJECT", "reason": reason_fail}

        # ------------------------------------------------------------------ #
        # Step 2: Context Profile & Fraud Scoring                             #
        # ------------------------------------------------------------------ #
        fraud_val = self.fraud_service.evaluate_risk(
            user_id=user_id,
            ip_address=ip_address,
            device_fingerprint=device_fingerprint,
            latitude=latitude,
            longitude=longitude
        )

        if fraud_val.action_required == "REJECT":
            await self._log_auth_failure(
                user_id=user_id, session_id=session_id, phrase_id=phrase_id,
                reason=f"Fraud engine auto-block: {fraud_val.reason}", ip=ip_address,
                risk_lvl=fraud_val.risk_level, risk_score=fraud_val.score
            )
            return {
                "authenticated": False,
                "decision": "REJECT",
                "reason": f"Access blocked by policy security engine: {fraud_val.reason}",
                "transaction_id": fraud_val.transaction_id,
            }

        # ------------------------------------------------------------------ #
        # Step 3: Fetch Biometric Anchors & Thresholds                       #
        # ------------------------------------------------------------------ #
        anchor_model = await self.anchor_repo.get_by_user_id(user_id)
        if not anchor_model:
            await self._log_auth_failure(
                user_id=user_id, session_id=session_id, phrase_id=phrase_id,
                reason="No voiceprint enrollment exists.", ip=ip_address
            )
            return {"authenticated": False, "decision": "REJECT", "reason": "No enrolled voice print profile found."}

        rolling_pool = await self.rolling_repo.get_by_user_id(user_id)
        threshold_model = await self.threshold_repo.get_or_create(user_id)

        # Calibrate current threshold
        ip_device_trusted = (fraud_val.score < 0.3)
        current_threshold = self.threshold_engine.calibrate_threshold(
            user_threshold=threshold_model,
            consec_failures=threshold_model.consecutive_failures,
            illness_active=threshold_model.illness_window_active,
            fraud_risk_level=fraud_val.risk_level,
            ip_device_trusted=ip_device_trusted
        )

        # ------------------------------------------------------------------ #
        # Step 4: Process Incoming Audio Sample                              #
        # ------------------------------------------------------------------ #
        temp_path = await save_upload_to_temp(audio_file)
        try:
            # Load file bytes to feed check interfaces
            with open(temp_path, "rb") as f:
                raw_bytes = f.read()

            # Preprocess / VAD / Noise suppressor
            proc = self.audio_pipeline.process_bytes(raw_bytes, for_enrollment=False)
            if not proc.accepted:
                await self._log_auth_failure(
                    user_id=user_id, session_id=session_id, phrase_id=phrase_id,
                    reason=f"Audio rejected: {proc.rejection_reason}", ip=ip_address,
                    risk_lvl=fraud_val.risk_level, risk_score=fraud_val.score
                )
                return {"authenticated": False, "decision": "REJECT", "reason": proc.rejection_reason}

            # ------------------------------------------------------------------ #
            # Step 5: Anti-Spoofing & Liveness Gating                            #
            # ------------------------------------------------------------------ #
            # Recheck liveness on file bytes
            spoof_res = self.antispoof_engine.analyze_liveness(raw_bytes, claims.get("phrase_text"))
            if spoof_res.is_spoof:
                # Flag velocity failure and increment lockouts
                threshold_model.consecutive_failures += 1
                await self.db.flush()

                await self._log_auth_failure(
                    user_id=user_id, session_id=session_id, phrase_id=phrase_id,
                    reason=spoof_res.rejection_reason or "Spoof signature matched", ip=ip_address,
                    liveliness_score=spoof_res.confidence, liveness_passed=False,
                    risk_lvl="CRITICAL", risk_score=max(0.9, fraud_val.score)
                )

                # Trigger emergency SIEM alerts for live presentation attack
                from app.fraud.kafka_publisher import get_kafka_publisher
                get_kafka_publisher().publish_event("BIOMETRIC_PRESENTATION_ATTACK", {
                    "user_id": str(user_id),
                    "session_id": session_id,
                    "confidence": spoof_res.confidence,
                    "type": spoof_res.method_detected,
                })

                return {"authenticated": False, "decision": "REJECT", "reason": spoof_res.rejection_reason}

            # ------------------------------------------------------------------ #
            # Step 6: Extract & BioHash Transform incoming voice                 #
            # ------------------------------------------------------------------ #
            import soundfile as sf
            denoised_temp = os.path.join(settings.TEMP_DIR, f"auth_{uuid.uuid4().hex}.wav")
            os.makedirs(settings.TEMP_DIR, exist_ok=True)
            sf.write(denoised_temp, proc.waveform, proc.sample_rate)

            try:
                emb = self.ecapa.get_embedding(denoised_temp)
            finally:
                cleanup_temp_file(denoised_temp)

            # Get BioHash challenge template mapping user's master secret key ID
            live_template = self.vault.generate_challenge_template(user_id, emb, anchor_model.key_id)

            # ------------------------------------------------------------------ #
            # Step 7: Decrypt Anchor & Rolling Templates                         #
            # ------------------------------------------------------------------ #
            anchor_vector = self.vault.decrypt_template(
                user_id=user_id,
                encrypted_embedding=anchor_model.encrypted_embedding,
                nonce=anchor_model.encryption_nonce,
                key_id=anchor_model.key_id
            )

            rolling_vectors = []
            for r in rolling_pool:
                try:
                    dec_r = self.vault.decrypt_template(
                        user_id=user_id,
                        encrypted_embedding=r.encrypted_embedding,
                        nonce=r.encryption_nonce,
                        key_id=r.key_id
                    )
                    rolling_vectors.append(dec_r)
                except Exception as ex:
                    logger.error("Failed to decrypt rolling embedding version at pool position %d: %s", r.pool_position, ex)

            # ------------------------------------------------------------------ #
            # Step 8: Multi-template Weighted Cosine Similarity Scoring           #
            # ------------------------------------------------------------------ #
            sim_score, anchor_sim, rolling_avg = self.scorer.compute_score(
                auth_template=live_template,
                anchor_template=anchor_vector,
                rolling_templates=rolling_vectors
            )

            # ------------------------------------------------------------------ #
            # Step 9: Decision bands classification                              #
            # ------------------------------------------------------------------ #
            band, description = classify_score(sim_score, current_threshold, settings.STEP_UP_LOWER_BOUND)

            # ------------------------------------------------------------------ #
            # Step 10: Commit results and schedule rolling updates               #
            # ------------------------------------------------------------------ #
            if band == DecisionBand.PASS:
                # Successful verification
                threshold_model.consecutive_failures = 0
                threshold_model.last_successful_auth_at = now_utc

                # Record database metrics
                await self.history_repo.create(
                    user_id=user_id, session_id=session_id, phrase_id=phrase_id,
                    attempt_number=threshold_model.consecutive_failures + 1,
                    liveness_score=spoof_res.confidence, is_liveness_passed=True,
                    similarity_score=sim_score, is_similarity_passed=True,
                    final_decision="PASS", risk_score=fraud_val.score, risk_level=fraud_val.risk_level,
                    ip_device_trusted=ip_device_trusted
                )

                # Check rolling update threshold criteria (>0.92 cosine sim score)
                # Adds candidate only if it matches structural templates safely without drift morphing
                if sim_score >= settings.ROLLING_UPDATE_TRIGGER_THRESHOLD:
                    # Update pool in DB
                    updated = await self.rolling_manager.update_pool(
                        session=self.db,
                        user_id=user_id,
                        auth_history_id=user_id,  # Link dummy or generate history ID
                        raw_embedding=emb,
                        auth_score=sim_score,
                        anchor_template=anchor_vector,
                        existing_rolling=rolling_pool
                    )
                    
                    if updated:
                        # Evict Redis cache to keep synchronised
                        await self.session_cache.invalidate_embeddings_cache(user_id)
                        await self.audit.log_event(
                            db=self.db,
                            event_type=AuditEventType.TEMPLATE_UPDATE,
                            user_id=user_id,
                            actor="SYSTEM_SCHEDULER",
                            ip_address="127.0.0.1",
                            details=f"Rolling embedding pool updated. Match score: {sim_score:.3f}",
                            payload={"score": sim_score}
                        )

                # Log event in Audit Vault
                await self.audit.log_event(
                    db=self.db,
                    event_type=AuditEventType.AUTH_PASS,
                    user_id=user_id,
                    actor="CUSTOMER",
                    ip_address=ip_address,
                    details=f"Voice authentication passed. Score: {sim_score:.3f} >= threshold: {current_threshold:.3f}",
                    payload={"score": sim_score, "threshold": current_threshold}
                )

                return {
                    "authenticated": True,
                    "decision": "PASS",
                    "similarity_score": round(sim_score, 3),
                    "reason": "OK",
                    "transaction_id": fraud_val.transaction_id,
                }

            elif band == DecisionBand.STEP_UP:
                # Soft failure, require OTP validation
                threshold_model.consecutive_failures += 1

                await self.history_repo.create(
                    user_id=user_id, session_id=session_id, phrase_id=phrase_id,
                    attempt_number=threshold_model.consecutive_failures,
                    liveness_score=spoof_res.confidence, is_liveness_passed=True,
                    similarity_score=sim_score, is_similarity_passed=False,
                    final_decision="STEP_UP", risk_score=fraud_val.score, risk_level=fraud_val.risk_level,
                    failure_reason=description, ip_device_trusted=ip_device_trusted
                )

                await self.audit.log_event(
                    db=self.db,
                    event_type=AuditEventType.AUTH_STEPUP,
                    user_id=user_id,
                    actor="CUSTOMER",
                    ip_address=ip_address,
                    details=f"Voice authentication soft-failed. Triggers Step-Up OTP. Score: {sim_score:.3f}",
                    payload={"score": sim_score, "threshold": current_threshold}
                )

                return {
                    "authenticated": False,
                    "decision": "STEP_UP",
                    "similarity_score": round(sim_score, 3),
                    "reason": "Verification score in Step-Up zone. Secondary authorization required.",
                    "transaction_id": fraud_val.transaction_id,
                }

            else:
                # Hard failure, access locked
                threshold_model.consecutive_failures += 1

                await self.history_repo.create(
                    user_id=user_id, session_id=session_id, phrase_id=phrase_id,
                    attempt_number=threshold_model.consecutive_failures,
                    liveness_score=spoof_res.confidence, is_liveness_passed=True,
                    similarity_score=sim_score, is_similarity_passed=False,
                    final_decision="FAIL", risk_score=fraud_val.score, risk_level=fraud_val.risk_level,
                    failure_reason=description, ip_device_trusted=ip_device_trusted
                )

                await self.audit.log_event(
                    db=self.db,
                    event_type=AuditEventType.AUTH_FAIL,
                    user_id=user_id,
                    actor="CUSTOMER",
                    ip_address=ip_address,
                    details=f"Voice verification hard failure. Score: {sim_score:.3f} < step_up zone limit",
                    payload={"score": sim_score, "threshold": current_threshold}
                )

                return {
                    "authenticated": False,
                    "decision": "FAIL",
                    "similarity_score": round(sim_score, 3),
                    "reason": f"Biometric speaker identity mismatch: {description}",
                    "transaction_id": fraud_val.transaction_id,
                }

        finally:
            cleanup_temp_file(temp_path)

    async def _log_auth_failure(
        self,
        user_id: uuid.UUID,
        session_id: str,
        phrase_id: str,
        reason: str,
        ip: str,
        liveliness_score: float = 1.0,
        liveness_passed: bool = True,
        risk_lvl: str = "LOW",
        risk_score: float = 0.05,
    ) -> None:
        """Helper to create error transaction entries without processing waveforms."""
        await self.history_repo.create(
            user_id=user_id, session_id=session_id, phrase_id=phrase_id,
            attempt_number=1,
            liveness_score=liveliness_score, is_liveness_passed=liveness_passed,
            similarity_score=0.0, is_similarity_passed=False,
            final_decision="FAIL", risk_score=risk_score, risk_level=risk_lvl,
            failure_reason=reason, ip_device_trusted=False
        )
        await self.audit.log_event(
            db=self.db,
            event_type=AuditEventType.AUTH_FAIL,
            user_id=user_id,
            actor="CUSTOMER",
            ip_address=ip,
            details=f"Auth failure: {reason}",
            payload={"session_id": session_id, "reason": reason}
        )
