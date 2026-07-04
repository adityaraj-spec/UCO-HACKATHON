"""
app/api/v1/endpoints/verify.py

POST /api/v1/verify

Verify a live audio sample against a user's enrolled voiceprint and return
the combined Layer1+Layer2 risk assessment.
"""

import uuid

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.database.session import get_db
from app.middleware.jwt_auth import get_current_user_id
from app.schemas.verification import VerificationResult
from app.services.verification_service import VerificationService
from app.utils.exceptions import (
    InvalidAudioFileError,
    UserNotFoundError,
    VoiceprintNotFoundError,
)

log = get_logger(__name__)

router = APIRouter()


@router.post(
    "/verify",
    response_model=VerificationResult,
    status_code=status.HTTP_200_OK,
    summary="Verify a live audio sample against an enrolled voiceprint",
    description=(
        "Accepts a user_id and a single audio recording. Generates an "
        "ECAPA-TDNN embedding for the recording, fetches the user's stored "
        "voiceprint, computes cosine similarity, applies the configured "
        "threshold (default 0.65), and returns the identity decision plus "
        "the combined risk assessment from the Risk Engine."
    ),
)
async def verify_speaker(
    file: UploadFile = File(..., description="Live audio recording to verify"),
    layer1_score: float = Form(
        0.0,
        ge=0.0,
        le=1.0,
        description=(
            "AI-voice-detection probability from Layer 1, in [0, 1]. "
            "Defaults to 0.0 if Layer 1 has not been run."
        ),
    ),
    user_id: uuid.UUID = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
) -> VerificationResult:
    service = VerificationService(session=db)

    try:
        return await service.verify(
            user_id=user_id, audio_file=file, layer1_score=layer1_score
        )
    except UserNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=exc.message
        ) from exc
    except VoiceprintNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=exc.message
        ) from exc
    except InvalidAudioFileError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=exc.message
        ) from exc
    except Exception as exc:  # noqa: BLE001
        log.exception("Unexpected error during verification")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred during verification.",
        ) from exc
