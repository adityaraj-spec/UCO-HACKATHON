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
from app.schemas.verification import VerificationResult
from app.services.verification_service import VerificationService
from app.utils.exceptions import (
    InvalidAudioFileError,
    UserNotFoundError,
    VoiceprintNotFoundError,
)

from app.api.v1.dependencies import rate_limiter, replay_protection

log = get_logger(__name__)

router = APIRouter()


@router.post(
    "/verify",
    response_model=VerificationResult,
    status_code=status.HTTP_200_OK,
    dependencies=[Depends(rate_limiter), Depends(replay_protection)],
    summary="Verify a live audio sample against an enrolled voiceprint",
    description=(
        "Accepts a user_id and a single audio recording. Generates an "
        "ECAPA-TDNN embedding for the recording, computes BioHash similarity "
        "against the user's stored voiceprint, applies s-norm identity "
        "decisioning, and returns that identity result plus the combined "
        "risk assessment from the Risk Engine."
    ),
)
async def verify_speaker(
    user_id: uuid.UUID = Form(..., description="UUID of the claimed user/account"),
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
    transaction_type: str | None = Form(
        None,
        description="Optional transaction type such as balance_inquiry, fund_transfer, add_payee, or limit_change.",
    ),
    transaction_amount: float | None = Form(
        None,
        ge=0.0,
        description="Optional transaction amount for money-moving actions.",
    ),
    expected_phrase: str | None = Form(
        None,
        description="Dynamic phrase the customer was asked to say for this transaction.",
    ),
    spoken_text: str | None = Form(
        None,
        description="ASR transcript of the customer's spoken phrase, if available.",
    ),
    db: AsyncSession = Depends(get_db),
) -> VerificationResult:
    service = VerificationService(session=db)

    try:
        return await service.verify(
            user_id=user_id,
            audio_file=file,
            layer1_score=layer1_score,
            expected_phrase=expected_phrase,
            spoken_text=spoken_text,
            transaction_type=transaction_type,
            transaction_amount=transaction_amount,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc
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


from app.schemas.verification import IdentificationResult, SpeakerMatch

@router.post(
    "/identify",
    response_model=IdentificationResult,
    status_code=status.HTTP_200_OK,
    dependencies=[Depends(rate_limiter), Depends(replay_protection)],
    summary="Identify speaker by searching across all enrolled voiceprints",
    description=(
        "Query the database using the in-memory FAISS index to find the "
        "top-k closest speaker matches for a given audio file."
    ),
)
async def identify_speaker(
    file: UploadFile = File(..., description="Audio recording to identify"),
    k: int = Form(5, description="Number of top matches to return"),
    db: AsyncSession = Depends(get_db),
) -> IdentificationResult:
    service = VerificationService(session=db)
    try:
        matches = await service.identify(audio_file=file, k=k)
        return IdentificationResult(matches=matches)
    except InvalidAudioFileError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=exc.message
        ) from exc
    except Exception as exc:
        log.exception("Unexpected error during speaker identification")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred during speaker identification.",
        ) from exc
