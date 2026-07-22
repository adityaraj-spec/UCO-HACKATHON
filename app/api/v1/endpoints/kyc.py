"""Dynamic sentence Voice KYC endpoints."""

import uuid

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.database.session import get_db
from app.schemas.kyc import (
    KYCAttemptResponse,
    KYCChallengeRequest,
    KYCChallengeResponse,
    KYCEnrolRequest,
    KYCEnrolResponse,
    KYCSessionStartRequest,
    KYCSessionStartResponse,
)
from app.services.kyc_enrollment_service import KYCEnrollmentService

router = APIRouter(prefix="/kyc")
log = get_logger(__name__)


@router.post("/session/start", response_model=KYCSessionStartResponse)
async def start_kyc_session(
    payload: KYCSessionStartRequest,
    db: AsyncSession = Depends(get_db),
) -> KYCSessionStartResponse:
    service = KYCEnrollmentService(db)
    try:
        kyc_session = await service.start_session(payload.phone_number)
        return KYCSessionStartResponse(session_id=kyc_session.session_id)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@router.post("/challenge", response_model=KYCChallengeResponse)
async def get_challenge_sentence(
    payload: KYCChallengeRequest,
    db: AsyncSession = Depends(get_db),
) -> KYCChallengeResponse:
    service = KYCEnrollmentService(db)
    try:
        challenge = await service.issue_challenge(payload.session_id, payload.attempt_no)
        return KYCChallengeResponse(
            sentence_text=challenge.sentence_text,
            expires_at=challenge.expires_at,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@router.post("/attempt", response_model=KYCAttemptResponse)
async def submit_kyc_attempt(
    session_id: uuid.UUID = Form(...),
    attempt_no: int = Form(..., ge=1, le=2),
    audio: UploadFile = File(...),
    asr_transcript: str | None = Form(None),
    db: AsyncSession = Depends(get_db),
) -> KYCAttemptResponse:
    service = KYCEnrollmentService(db)
    try:
        success, match_score, message = await service.submit_attempt(
            session_id=session_id,
            attempt_no=attempt_no,
            audio=audio,
            asr_transcript=asr_transcript,
        )
        return KYCAttemptResponse(
            success=success,
            asr_match_score=match_score,
            message=message,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@router.post("/enrol", response_model=KYCEnrolResponse)
async def enrol_kyc_session(
    payload: KYCEnrolRequest,
    db: AsyncSession = Depends(get_db),
) -> KYCEnrolResponse:
    service = KYCEnrollmentService(db)
    try:
        salt_id, message = await service.enrol(payload.session_id)
        return KYCEnrolResponse(success=True, salt_id=salt_id, message=message)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as exc:
        log.exception("Unexpected error during KYC enrolment: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Enrolment failed: {exc}",
        ) from exc

