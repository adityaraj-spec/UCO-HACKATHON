"""
app/api/v1/endpoints/history.py

GET /api/v1/verification-history/{id}
GET /api/v1/risk-history/{id}

History endpoints used by the Streamlit "History" and "Risk Dashboard" pages.
"""

import uuid

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.database.session import get_db
from app.middleware.jwt_auth import get_current_user_id
from app.repositories.risk_repository import RiskLogRepository
from app.repositories.user_repository import UserRepository
from app.repositories.verification_repository import VerificationLogRepository
from app.schemas.risk import RiskLogRead
from app.schemas.verification import VerificationLogRead
from app.utils.exceptions import UserNotFoundError

log = get_logger(__name__)

router = APIRouter()


@router.get(
    "/verification-history/me",
    response_model=list[VerificationLogRead],
    status_code=status.HTTP_200_OK,
    summary="Get the authenticated customer's speaker-verification history",
)
async def get_my_verification_history(
    user_id: uuid.UUID = Depends(get_current_user_id),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
) -> list[VerificationLogRead]:
    repo = VerificationLogRepository(db)
    logs = await repo.list_by_user_id(user_id, limit=limit, offset=offset)
    return [VerificationLogRead.model_validate(log_entry) for log_entry in logs]


@router.get(
    "/risk-history/me",
    response_model=list[RiskLogRead],
    status_code=status.HTTP_200_OK,
    summary="Get the authenticated customer's risk-engine history",
)
async def get_my_risk_history(
    user_id: uuid.UUID = Depends(get_current_user_id),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
) -> list[RiskLogRead]:
    repo = RiskLogRepository(db)
    logs = await repo.list_by_user_id(user_id, limit=limit, offset=offset)
    return [RiskLogRead.model_validate(log_entry) for log_entry in logs]


@router.get(
    "/verification-history/{user_id}",
    response_model=list[VerificationLogRead],
    status_code=status.HTTP_200_OK,
    summary="Get a user's speaker-verification history",
)
async def get_verification_history(
    user_id: uuid.UUID,
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
) -> list[VerificationLogRead]:
    user_repo = UserRepository(db)
    if not await user_repo.exists(user_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=UserNotFoundError(str(user_id)).message,
        )

    repo = VerificationLogRepository(db)
    logs = await repo.list_by_user_id(user_id, limit=limit, offset=offset)
    return [VerificationLogRead.model_validate(log_entry) for log_entry in logs]


@router.get(
    "/risk-history/{user_id}",
    response_model=list[RiskLogRead],
    status_code=status.HTTP_200_OK,
    summary="Get a user's risk-engine evaluation history",
)
async def get_risk_history(
    user_id: uuid.UUID,
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
) -> list[RiskLogRead]:
    user_repo = UserRepository(db)
    if not await user_repo.exists(user_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=UserNotFoundError(str(user_id)).message,
        )

    repo = RiskLogRepository(db)
    logs = await repo.list_by_user_id(user_id, limit=limit, offset=offset)
    return [RiskLogRead.model_validate(log_entry) for log_entry in logs]
