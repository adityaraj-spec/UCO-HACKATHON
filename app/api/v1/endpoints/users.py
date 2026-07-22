"""
app/api/v1/endpoints/users.py

User management endpoints:

    POST   /api/v1/users           - create a new enrollable user
    GET    /api/v1/users/{id}       - fetch a user and enrollment status
"""

import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.logging import get_logger
from app.database.session import get_db
from app.models.user import User
from app.repositories.voiceprint_repository import VoiceprintRepository
from app.repositories.user_repository import UserRepository
from app.schemas.user import UserCreate, UserDetailRead, UserRead
from app.utils.exceptions import UserAlreadyExistsError, UserNotFoundError

log = get_logger(__name__)

router = APIRouter()


@router.post(
    "/users",
    response_model=UserRead,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new user",
    description=(
        "Creates a new user record. A user must exist before they can be "
        "enrolled (POST /api/v1/enroll) or verified (POST /api/v1/verify)."
    ),
)
async def create_user(
    payload: UserCreate, db: AsyncSession = Depends(get_db)
) -> UserRead:
    repo = UserRepository(db)

    # Normalize email
    clean_email = payload.email.strip().lower()

    existing = await repo.get_by_email(clean_email)
    if existing is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"User with email '{payload.email}' already exists.",
        )

    try:
        user = await repo.create(name=payload.name, email=clean_email)
        await db.commit()
        return UserRead.model_validate(user)
    except IntegrityError as exc:
        await db.rollback()
        log.warning("Duplicate user creation attempted for email: %s", payload.email)
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"User with email '{payload.email}' already exists.",
        ) from exc
    except Exception as exc:  # noqa: BLE001
        await db.rollback()
        log.exception("Failed to create user")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create user: {exc}",
        ) from exc


@router.get(
    "/users/{user_id}",
    response_model=UserDetailRead,
    status_code=status.HTTP_200_OK,
    summary="Get a user and their enrollment status",
)
async def get_user(
    user_id: uuid.UUID, db: AsyncSession = Depends(get_db)
) -> UserDetailRead:
    user_repo = UserRepository(db)
    voiceprint_repo = VoiceprintRepository(db)
    user: User | None = await user_repo.get_by_id(user_id)

    if user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=UserNotFoundError(str(user_id)).message,
        )

    voiceprint = await voiceprint_repo.get_by_user_id(user_id)

    return UserDetailRead(
        id=user.id,
        name=user.name,
        email=user.email,
        created_at=user.created_at,
        is_enrolled=voiceprint is not None,
        recording_count=voiceprint.recording_count if voiceprint else 0,
        embedding_dimension=len(voiceprint.embedding) if (voiceprint and voiceprint.embedding) else None,
        voiceprint_updated_at=voiceprint.updated_at if voiceprint else None,
    )


@router.delete(
    "/users/{user_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Hard deletion cascade for compliance (GDPR/ISO 27001)",
)
async def delete_user(
    user_id: uuid.UUID, db: AsyncSession = Depends(get_db)
):
    """Deletes user, voiceprint, purge FAISS index entry, and clears Redis hashes."""
    repo = UserRepository(db)
    user = await repo.get_by_id(user_id)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=UserNotFoundError(str(user_id)).message,
        )

    await db.delete(user)
    await db.commit()

    # Rebuild FAISS index
    from app.services.faiss_service import faiss_service
    await faiss_service.sync_with_db(db)
