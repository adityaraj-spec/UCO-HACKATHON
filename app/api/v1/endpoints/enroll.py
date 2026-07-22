"""
app/api/v1/endpoints/enroll.py

POST /api/v1/enroll

Enroll a user's voiceprint from multiple WAV recordings.
"""

import uuid

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.database.session import get_db
from app.schemas.enrollment import EnrollmentResponse
from app.services.enrollment_service import EnrollmentService
from app.utils.exceptions import (
    EnrollmentRejectedError,
    InsufficientRecordingsError,
    InvalidAudioFileError,
    UserNotFoundError,
)

from app.api.v1.dependencies import rate_limiter, replay_protection

log = get_logger(__name__)

router = APIRouter()


@router.post(
    "/enroll",
    response_model=EnrollmentResponse,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(rate_limiter), Depends(replay_protection)],
    summary="Enroll a user's voiceprint",
    description=(
        "Accepts a user_id and multiple WAV recordings (30-50 utterances "
        "recommended). Generates ECAPA-TDNN embeddings for each recording, "
        "averages them into a single voiceprint, and stores the result in "
        "PostgreSQL via pgvector."
    ),
)
async def enroll_user_voiceprint(
    user_id: uuid.UUID = Form(..., description="UUID of the user to enroll"),
    files: list[UploadFile] = File(
        ..., description="One or more WAV/FLAC/MP3 recordings of the user's voice"
    ),
    channel: str = Form("DIRECT_API", description="Enrollment channel: DIRECT_API, VIDEO_KYC, SECURE_MAILER, or BRANCH."),
    biometric_consent_confirmed: bool = Form(False, description="Explicit biometric consent confirmation."),
    identity_confirmed: bool = Form(False, description="Officer-confirmed identity check for branch/video KYC flows."),
    authenticated: bool = Form(False, description="Whether app login/MPIN authentication succeeded before recording."),
    otp_verified: bool = Form(False, description="Whether a fresh OTP was verified before enrollment."),
    device_id: str | None = Form(None, description="Stable client device identifier when available."),
    branch_officer_id: str | None = Form(None, description="Branch officer id for branch-assisted enrollment."),
    db: AsyncSession = Depends(get_db),
) -> EnrollmentResponse:
    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one audio file must be provided for enrollment.",
        )

    service = EnrollmentService(session=db)

    try:
        return await service.enroll(
            user_id=user_id,
            audio_files=files,
            channel=channel,
            biometric_consent_confirmed=biometric_consent_confirmed,
            identity_confirmed=identity_confirmed,
            authenticated=authenticated,
            otp_verified=otp_verified,
            device_id=device_id,
            branch_officer_id=branch_officer_id,
        )
    except UserNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=exc.message
        ) from exc
    except InsufficientRecordingsError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=exc.message
        ) from exc
    except EnrollmentRejectedError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=exc.message
        ) from exc
    except InvalidAudioFileError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=exc.message
        ) from exc
    except Exception as exc:  # noqa: BLE001
        log.exception("Unexpected error during enrollment")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred during enrollment.",
        ) from exc
