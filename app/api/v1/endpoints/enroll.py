"""
app/api/v1/endpoints/enroll.py

POST /api/v1/enroll

Enroll a user's voiceprint from multiple WAV recordings.
"""

import uuid

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.database.session import get_db
from app.middleware.jwt_auth import get_current_user_id
from app.middleware.rate_limiter import limiter
from app.schemas.enrollment import EnrollmentResponse
from app.services.enrollment_service import EnrollmentService
from app.utils.exceptions import (
    InsufficientRecordingsError,
    InvalidAudioFileError,
    UserNotFoundError,
)

log = get_logger(__name__)

router = APIRouter()


@router.post(
    "/enroll",
    response_model=EnrollmentResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Enroll a user's voiceprint",
    description=(
        "Accepts a user_id and multiple WAV recordings (30-50 utterances "
        "recommended). Generates ECAPA-TDNN embeddings for each recording, "
        "averages them into a single voiceprint, and stores the result in "
        "PostgreSQL via pgvector."
    ),
)
@limiter.limit("3/hour")
async def enroll_user_voiceprint(
    request: Request,
    files: list[UploadFile] = File(
        ..., description="One or more WAV/FLAC/MP3 recordings of the user's voice"
    ),
    user_id: uuid.UUID = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
) -> EnrollmentResponse:
    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one audio file must be provided for enrollment.",
        )

    service = EnrollmentService(session=db)

    try:
        return await service.enroll(user_id=user_id, audio_files=files)
    except UserNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=exc.message
        ) from exc
    except InsufficientRecordingsError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=exc.message
        ) from exc
    except InvalidAudioFileError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=exc.message
        ) from exc
    except PermissionError as exc:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)
        ) from exc
    except Exception as exc:  # noqa: BLE001
        log.exception("Unexpected error during enrollment")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Enrollment failed: {exc}",
        ) from exc
