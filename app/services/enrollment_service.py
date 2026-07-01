"""
app/services/enrollment_service.py

Voice enrollment service.

Enforces Bank-grade voice verification enrollment requirements:
  - DPDP consent verification gate
  - Session-based sample collection (5 samples required)
  - Full noise, clipping, DC offset, and SNR validation per sample
  - Audio trimming & noise suppression preprocessing
  - Detached anchor calculation
  - BioHash transformation & AES-256-GCM vault encryption
  - Automatic temp file and raw audio deletion
  - Audit logging configuration
"""

from __future__ import annotations

import logging
import os
import uuid
import numpy as np
from fastapi import UploadFile

from app.audit.event_types import AuditEventType
from app.audit.vault import get_audit_vault
from app.biometrics.template_vault import get_template_vault
from app.consent.consent_service import get_consent_service
from app.core.config import get_settings
from app.database.session import AsyncSession
from app.ml.ecapa_service import get_ecapa_service
from app.models.layer2.enrollment import EnrollmentSession
from app.models.layer2.voice_metadata import VoiceMetadata
from app.models.layer2.user_threshold import UserThreshold
from app.models.layer2.rolling_embedding import RollingEmbedding
from app.preprocessing.audio_pipeline import get_audio_pipeline
from app.repositories.layer2.enrollment_repo import EnrollmentSessionRepository
from app.repositories.layer2.anchor_embedding_repo import AnchorEmbeddingRepository
from app.repositories.layer2.threshold_repo import UserThresholdRepository
from app.utils.audio_utils import cleanup_temp_file, save_upload_to_temp

logger = logging.getLogger(__name__)
settings = get_settings()


class EnrollmentService:
    """Manages multi-sample user voice enrollment with compliance and quality verification gates."""

    def __init__(self, session: AsyncSession) -> None:
        self.db = session
        self.enrollment_repo = EnrollmentSessionRepository(session)
        self.anchor_repo = AnchorEmbeddingRepository(session)
        self.threshold_repo = UserThresholdRepository(session)
        self.consent_service = get_consent_service()
        self.audio_pipeline = get_audio_pipeline()
        self.ecapa = get_ecapa_service()
        self.vault = get_template_vault()
        self.audit = get_audit_vault()

    async def create_enrollment_session(
        self,
        user_id: uuid.UUID,
        ip_address: str,
        user_agent: str,
    ) -> EnrollmentSession:
        """
        Start a new enrollment session if DPDP consent is active.
        """
        # Step 1: Pre-requisite: DPDP Consent check
        has_consent = await self.consent_service.verify_active_consent(self.db, user_id)
        if not has_consent:
            raise PermissionError("DPDP consent is required before initializing biometric voice enrollment.")

        # Step 2: Clear any active stale session and create a new one
        active = await self.enrollment_repo.get_active_session(user_id)
        if active:
            active.status = "REVOKED"
            logger.info("Revoked previous in-progress enrollment session for user: %s", user_id)

        sess = await self.enrollment_repo.create(user_id)
        await self.audit.log_event(
            db=self.db,
            event_type=AuditEventType.ENROLLMENT_START,
            user_id=user_id,
            actor="CUSTOMER",
            ip_address=ip_address,
            details=f"Enrollment session {sess.id} started.",
            payload={"session_id": str(sess.id)}
        )
        return sess

    async def submit_sample(
        self,
        user_id: uuid.UUID,
        session_id: uuid.UUID,
        audio_file: UploadFile,
        ip_address: str,
    ) -> dict:
        """
        Process and validate a single enrollment sample.

        Does NOT store embeddings to database directly. Saves raw processed samples
        temporarily inside the temporary filesystem/cache, bound to session context.
        """
        # Step 1: Query session
        sess = await self.enrollment_repo.get_by_id(session_id)
        if not sess or sess.user_id != user_id or sess.status != "IN_PROGRESS":
            raise ValueError("Invalid or inactive enrollment session.")

        temp_path = None
        try:
            # Step 2: Save upload payload locally
            temp_path = await save_upload_to_temp(audio_file)

            # Step 3: Run master audio preprocessing & quality pipeline
            proc = self.audio_pipeline.process_file(temp_path, for_enrollment=True)
            if not proc.accepted:
                raise ValueError(f"Quality validation failed: {proc.rejection_reason}")

            # Step 4: Extract ECAPA embedding
            # (Note: we bypass normalizer in ecapa_service if already denoised/16k)
            # Create a temporary path for the denoised waveform because ecapa_service expects paths
            import soundfile as sf
            denoised_temp = os.path.join(settings.TEMP_DIR, f"denoised_{uuid.uuid4().hex}.wav")
            os.makedirs(settings.TEMP_DIR, exist_ok=True)
            sf.write(denoised_temp, proc.waveform, proc.sample_rate)

            try:
                emb = self.ecapa.get_embedding(denoised_temp)
            finally:
                cleanup_temp_file(denoised_temp)

            # Step 5: Save processed sample wave vectors and SNRs temporarily
            # For this banking system, we accumulate sample results inside the Session
            # object cache or a temp file directory bound to the session.
            records_dir = os.path.join(settings.TEMP_DIR, f"sess_samples_{session_id}")
            os.makedirs(records_dir, exist_ok=True)

            sample_idx = sess.samples_submitted
            np.save(os.path.join(records_dir, f"sample_{sample_idx}.npy"), emb)

            # Save quality metadata for aggregations later
            meta_path = os.path.join(records_dir, f"meta_{sample_idx}.json")
            import json
            with open(meta_path, "w") as f:
                json.dump({
                    "snr_db": proc.quality.snr_db,
                    "voice_ratio": proc.quality.voice_ratio,
                    "duration": proc.processed_duration_seconds,
                    "quality_score": proc.quality.quality_score,
                }, f)

            sess.samples_submitted += 1
            await self.db.flush()

            # Record event in audit vault
            await self.audit.log_event(
                db=self.db,
                event_type=AuditEventType.ENROLLMENT_SAMPLE_SUBMIT,
                user_id=user_id,
                actor="CUSTOMER",
                ip_address=ip_address,
                details=f"Enrollment sample {sess.samples_submitted}/5 submitted.",
                payload={
                    "session_id": str(session_id),
                    "sample_number": sess.samples_submitted,
                    "snr_db": proc.quality.snr_db,
                    "quality_score": proc.quality.quality_score,
                }
            )

            is_complete = sess.samples_submitted >= sess.samples_required

            return {
                "success": True,
                "samples_submitted": sess.samples_submitted,
                "samples_required": sess.samples_required,
                "is_complete": is_complete,
                "quality_report": {
                    "snr_db": round(proc.quality.snr_db, 2),
                    "voice_ratio": round(proc.quality.voice_ratio, 2),
                    "quality_score": proc.quality.quality_score,
                    "duration_seconds": round(proc.processed_duration_seconds, 2),
                }
            }

        finally:
            cleanup_temp_file(temp_path)

    async def finalize_enrollment(
        self,
        user_id: uuid.UUID,
        session_id: uuid.UUID,
        ip_address: str,
    ) -> dict:
        """
        Aggregate the 5 collected samples, compute anchor, BioHash & GCM encrypt templates.
        """
        # Step 1: Query session
        sess = await self.enrollment_repo.get_by_id(session_id)
        if not sess or sess.user_id != user_id or sess.status != "IN_PROGRESS":
            raise ValueError("Invalid or inactive enrollment session.")

        if sess.samples_submitted < sess.samples_required:
            raise ValueError(f"Insufficient samples collected: {sess.samples_submitted}/{sess.samples_required}")

        records_dir = os.path.join(settings.TEMP_DIR, f"sess_samples_{session_id}")
        if not os.path.exists(records_dir):
            raise FileNotFoundError("Enrollment session files missing or deleted.")

        try:
            # Step 2: Load intermediate embedding vectors
            embeddings = []
            snr_vals = []
            vr_vals = []
            scores = []
            dur_vals = []

            import json
            for i in range(sess.samples_required):
                emb_file = os.path.join(records_dir, f"sample_{i}.npy")
                meta_file = os.path.join(records_dir, f"meta_{i}.json")

                if not os.path.exists(emb_file):
                    raise FileNotFoundError(f"Missing sample data for index {i}.")

                embeddings.append(np.load(emb_file))
                with open(meta_file) as f:
                    meta = json.load(f)
                    snr_vals.append(meta["snr_db"])
                    vr_vals.append(meta["voice_ratio"])
                    scores.append(meta["quality_score"])
                    dur_vals.append(meta["duration"])

            # Step 3: Compute mean anchor embedding vector
            anchor_emb = np.mean(embeddings, axis=0)

            # Step 4: Apply BioHash transformation and GCM encryption via vault
            enc_blob, nonce, val_hash, key_used = self.vault.generate_template(user_id, anchor_emb)

            # Step 5: Save database models
            # Add to Anchor table
            await self.anchor_repo.create(
                user_id=user_id,
                encrypted_embedding=enc_blob,
                encryption_nonce=nonce,
                embedding_hash=val_hash,
                key_id=key_used,
            )

            # Initialize rolling pool with 3 copies of the anchor (as baseline)
            # This ensures similarity scores can fall back safely to ensemble modes
            for pos in range(3):
                # Apply same encryption (unique nonces generated during each vault call)
                r_enc_blob, r_nonce, r_val_hash, r_key_used = self.vault.generate_template(user_id, anchor_emb)
                
                roll_entry = RollingEmbedding(
                    user_id=user_id,
                    encrypted_embedding=r_enc_blob,
                    encryption_nonce=r_nonce,
                    embedding_hash=r_val_hash,
                    key_id=r_key_used,
                    auth_similarity_score=1.0,
                    cosine_distance_from_anchor=0.0,
                    pool_position=pos,
                    is_active=True,
                )
                self.db.add(roll_entry)

            # Step 6: Create or update User Threshold defaults
            ut = await self.threshold_repo.get_or_create(user_id)
            ut.threshold_baseline = settings.SIMILARITY_THRESHOLD
            ut.threshold_current = settings.SIMILARITY_THRESHOLD
            ut.consecutive_failures = 0
            ut.enrollment_quality_score = float(np.mean(scores))
            self.db.add(ut)

            # Save aggregations in VoiceMetadata profiles
            avg_snr = float(np.mean(snr_vals))
            q_level = "EXCELLENT" if avg_snr > 30 else ("HIGH" if avg_snr > 23 else "MEDIUM")

            v_meta = VoiceMetadata(
                user_id=user_id,
                enrollment_session_id=session_id,
                sample_count=sess.samples_required,
                avg_snr_db=avg_snr,
                min_snr_db=float(np.min(snr_vals)),
                max_snr_db=float(np.max(snr_vals)),
                avg_voice_ratio=float(np.mean(vr_vals)),
                avg_duration_seconds=float(np.mean(dur_vals)),
                avg_quality_score=float(np.mean(scores)),
                enrollment_quality=q_level,
                enrolled_via="FASTAPI_L2_SERVICE",
                sample_rate_used=16000,
                embedding_model="ECAPA-TDNN",
            )
            self.db.add(v_meta)

            # Step 7: Update session status
            sess.status = "COMPLETED"
            
            # Step 8: Update standard FAISS vector index
            # Retrieve decrypted projection to populate in-memory searching
            decrypted_anchor = self.vault.decrypt_template(user_id, enc_blob, nonce, key_used)
            from app.search.faiss_index import get_faiss_manager
            faiss_manager = get_faiss_manager()
            faiss_manager.add_vector(str(user_id), decrypted_anchor)

            # Step 9: Commit changes
            import shutil
            shutil.rmtree(records_dir, ignore_errors=True)

            # Audit event log
            await self.audit.log_event(
                db=self.db,
                event_type=AuditEventType.ENROLLMENT_COMPLETE,
                user_id=user_id,
                actor="CUSTOMER",
                ip_address=ip_address,
                details="Enrollment fully aggregated and saved. Purged raw files.",
                payload={"session_id": str(session_id)}
            )

            return {
                "success": True,
                "user_id": user_id,
                "quality_level": q_level,
                "message": "Enrollment completed successfully. Cancelable voice prints generated."
            }

        except Exception as e:
            logger.exception("Finalize enrollment failed for user: %s", user_id)
            raise e
