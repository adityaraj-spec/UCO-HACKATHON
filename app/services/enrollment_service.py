"""
app/services/enrollment_service.py

Business logic for the voice enrollment workflow:

    multiple WAV uploads -> temp storage -> ECAPA embeddings
        -> average -> voiceprint stored in PostgreSQL (pgvector)

Used by POST /api/v1/enroll.
"""

import uuid

from fastapi import UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.logging import get_logger
from app.ml.ecapa_service import ECAPAService, get_ecapa_service
from app.repositories.user_repository import UserRepository
from app.repositories.voiceprint_repository import VoiceprintRepository
from app.schemas.enrollment import EnrollmentResponse
from app.services.enrollment_access import (
    CHANNEL_DIRECT_API,
    EnrollmentAccessContext,
    EnrollmentAccessService,
)
from app.services.layer1_liveness_service import Layer1LivenessService
from app.utils.audio_quality import validate_enrollment_audio_quality
from app.utils.audio_utils import cleanup_temp_file, load_waveform, save_upload_to_temp
from app.utils.exceptions import EnrollmentRejectedError, InsufficientRecordingsError, UserNotFoundError

settings = get_settings()
log = get_logger(__name__)

LAYER1_LIVENESS_DISABLED_WARNING = (
    "LAYER1 LIVENESS CHECK DISABLED — DEV/DEMO MODE ONLY, DO NOT USE IN PRODUCTION."
)


class EnrollmentService:
    """Orchestrates the enrollment pipeline: audio -> embeddings -> voiceprint."""

    def __init__(
        self,
        session: AsyncSession,
        ecapa_service: ECAPAService | None = None,
        liveness_service: Layer1LivenessService | None = None,
        access_service: EnrollmentAccessService | None = None,
    ) -> None:
        self.session = session
        self.user_repo = UserRepository(session)
        self.voiceprint_repo = VoiceprintRepository(session)
        self.ecapa_service = ecapa_service or get_ecapa_service()
        self.liveness_service = liveness_service or Layer1LivenessService()
        self.access_service = access_service or EnrollmentAccessService()

    async def enroll(
        self,
        user_id: uuid.UUID,
        audio_files: list[UploadFile],
        channel: str = CHANNEL_DIRECT_API,
        biometric_consent_confirmed: bool = False,
        identity_confirmed: bool = False,
        authenticated: bool = False,
        otp_verified: bool = False,
        device_id: str | None = None,
        branch_officer_id: str | None = None,
    ) -> EnrollmentResponse:
        """
        Enroll (or re-enroll) a user's voiceprint from multiple recordings.

        Steps:
            1. Verify the user exists.
            2. Persist each upload to a temporary WAV file.
            3. Extract and average ECAPA-TDNN embeddings across all files.
            4. Validate that enough recordings were successfully processed.
            5. Upsert the resulting voiceprint into PostgreSQL (pgvector).
            6. Clean up temporary files.

        Raises:
            UserNotFoundError: if `user_id` does not correspond to a user.
            InsufficientRecordingsError: if fewer than
                `settings.MIN_ENROLLMENT_RECORDINGS` recordings were
                successfully processed.
        """
        user = await self.user_repo.get_by_id(user_id)
        if user is None:
            raise UserNotFoundError(str(user_id))

        existing_voiceprint = (
            await self.voiceprint_repo.get_by_user_id(user_id)
            if device_id
            else None
        )
        access_context = EnrollmentAccessContext(
            channel=channel,
            biometric_consent_confirmed=biometric_consent_confirmed,
            identity_confirmed=identity_confirmed,
            authenticated=authenticated,
            otp_verified=otp_verified,
            device_id=device_id,
            branch_officer_id=branch_officer_id,
        )
        self.access_service.validate(
            user_id=user_id,
            has_existing_voiceprint=existing_voiceprint is not None,
            context=access_context,
        )

        temp_paths: list[str] = []
        try:
            for upload in audio_files:
                temp_path = await save_upload_to_temp(upload)
                temp_paths.append(temp_path)
                waveform, sample_rate = load_waveform(temp_path)
                validate_enrollment_audio_quality(waveform, sample_rate)
                if settings.LAYER1_LIVENESS_CHECK_ENABLED:
                    ai_probability = self.liveness_service.predict_ai_probability(temp_path)
                    if ai_probability > settings.LAYER1_FRAUD_THRESHOLD:
                        raise EnrollmentRejectedError(
                            f"Layer 1 AI-voice probability {ai_probability:.4f} exceeds "
                            f"threshold {settings.LAYER1_FRAUD_THRESHOLD:.4f}."
                        )
                else:
                    log.warning(LAYER1_LIVENESS_DISABLED_WARNING)

            log.info(
                f"Starting enrollment for user_id={user_id} "
                f"with {len(temp_paths)} uploaded recordings"
            )

            averaged_embedding, processed_count = self.ecapa_service.enroll_user(
                temp_paths
            )

            if processed_count < settings.MIN_ENROLLMENT_RECORDINGS:
                raise InsufficientRecordingsError(
                    provided=processed_count,
                    required=settings.MIN_ENROLLMENT_RECORDINGS,
                )

            voiceprint = await self.voiceprint_repo.upsert(
                user_id=user_id,
                embedding=averaged_embedding,
                recording_count=processed_count,
            )
            await self.session.commit()
            self.access_service.remember_device(user_id, device_id)

            # Update FAISS index
            try:
                from app.services.faiss_service import faiss_service
                faiss_service.add_or_update(user_id, voiceprint.embedding)
            except Exception as exc:
                log.error(f"Failed to update FAISS index for user_id={user_id}: {exc}")

            log.info(
                f"Enrollment complete for user_id={user_id}: "
                f"{processed_count} recordings, "
                f"embedding_dimension={len(voiceprint.embedding)}"
            )

            return EnrollmentResponse(
                success=True,
                user_id=user_id,
                recording_count=processed_count,
                embedding_dimension=len(voiceprint.embedding),
                message=(
                    f"Voiceprint successfully created from "
                    f"{processed_count} recording(s)."
                ),
            )
        except (UserNotFoundError, InsufficientRecordingsError, EnrollmentRejectedError):
            await self.session.rollback()
            raise
        except Exception:
            await self.session.rollback()
            log.exception(f"Enrollment failed for user_id={user_id}")
            raise
        finally:
            for path in temp_paths:
                cleanup_temp_file(path)
