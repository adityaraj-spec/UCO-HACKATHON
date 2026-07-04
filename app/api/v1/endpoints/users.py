"""
app/api/v1/endpoints/users.py

Banking identity authentication endpoints.
"""

from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.audit.event_types import AuditEventType
from app.audit.vault import get_audit_vault

from app.cache.redis_client import get_redis_client
from app.core.config import get_settings
from app.core.logging import get_logger
from app.database.session import get_db
from app.middleware.jwt_auth import (
    create_access_token,
    get_current_user_claims,
    get_current_user_id,
)
from app.middleware.rate_limiter import limiter
from app.middleware.rbac import RoleChecker
from app.models.user import User
from app.repositories.layer2.anchor_embedding_repo import AnchorEmbeddingRepository
from app.repositories.user_repository import UserRepository
from app.schemas.user import (
    AccountNumberRequest,
    BankCustomerCreate,
    MPINChangeRequest,
    MPINLoginRequest,
    MPINLoginResponse,
    UserDetailRead,
    UserPublicRead,
    UserRead,
    mask_phone,
)
from app.utils.exceptions import UserAlreadyExistsError, UserNotFoundError
from app.utils.mpin_hasher import hash_mpin, verify_mpin

log = get_logger(__name__)
settings = get_settings()

router = APIRouter()
allow_staff = RoleChecker(["AGENT", "ADMIN", "STAFF"])


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _as_aware(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


async def _is_enrolled(db: AsyncSession, user_id: uuid.UUID) -> bool:
    anchor = await AnchorEmbeddingRepository(db).get_by_user_id(user_id)
    return anchor is not None


def _public_user(user: User, is_enrolled: bool) -> UserPublicRead:
    return UserPublicRead(
        account_number=user.account_number,
        full_name=user.full_name,
        phone_number=mask_phone(user.phone_number),
        branch_code=user.branch_code,
        account_type=user.account_type,
        kyc_status=user.kyc_status,
        is_enrolled=is_enrolled,
        last_login_at=user.last_login_at,
    )


async def _reject_replayed_login(payload: MPINLoginRequest) -> None:
    if payload.request_nonce is None:
        return
    if payload.request_timestamp is not None:
        timestamp = _as_aware(payload.request_timestamp)
        assert timestamp is not None
        age = abs((_now_utc() - timestamp).total_seconds())
        if age > settings.LOGIN_REPLAY_WINDOW_SECONDS:
            raise HTTPException(status_code=401, detail="Login request timestamp expired.")

    redis = await get_redis_client()
    digest = hashlib.sha256(
        f"{payload.account_number}:{payload.request_nonce}".encode()
    ).hexdigest()
    key = f"login_nonce:{digest}"
    if await redis.get(key):
        raise HTTPException(status_code=401, detail="Replayed login request rejected.")
    await redis.set(key, "1", expire_seconds=settings.LOGIN_REPLAY_WINDOW_SECONDS)


def _device_hash(device_fingerprint: str) -> str:
    return hashlib.sha256(device_fingerprint.encode()).hexdigest()


@router.post(
    "/auth/register",
    response_model=UserRead,
    status_code=status.HTTP_201_CREATED,
    tags=["Banking Auth"],
)
@limiter.limit("5/minute")
async def register_customer(
    request: Request,
    payload: BankCustomerCreate,
    db: AsyncSession = Depends(get_db),
) -> UserRead:
    repo = UserRepository(db)

    if await repo.get_by_account_number(payload.account_number):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=UserAlreadyExistsError(payload.account_number).message,
        )
    if await repo.get_by_phone(payload.phone_number):
        raise HTTPException(status_code=409, detail="Phone number is already registered.")

    try:
        user = await repo.create_bank_customer(
            account_number=payload.account_number,
            mpin_hash=hash_mpin(payload.mpin),
            full_name=payload.full_name,
            phone_number=payload.phone_number,
            dob=payload.date_of_birth,
            branch_code=payload.branch_code,
            account_type=payload.account_type,
            email=str(payload.email) if payload.email else None,
        )
        await db.commit()
        return UserRead.model_validate(user)
    except Exception as exc:  # noqa: BLE001
        await db.rollback()
        log.exception("Failed to register bank customer")
        raise HTTPException(status_code=500, detail="Failed to register customer.") from exc


@router.post(
    "/auth/login",
    response_model=MPINLoginResponse,
    status_code=status.HTTP_200_OK,
    tags=["Banking Auth"],
)
@limiter.limit("10/minute")
async def login_with_mpin(
    payload: MPINLoginRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> MPINLoginResponse:
    await _reject_replayed_login(payload)

    repo = UserRepository(db)
    user = await repo.get_by_account_number(payload.account_number)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid account number or MPIN.",
        )

    locked_until = _as_aware(user.mpin_locked_until)
    if locked_until and locked_until > _now_utc():
        raise HTTPException(status_code=423, detail="Account temporarily locked. Try again later.")

    if not user.is_active:
        raise HTTPException(status_code=403, detail="Account frozen - contact branch.")
    if user.kyc_status == "REJECTED":
        raise HTTPException(status_code=403, detail="KYC rejected - contact branch.")

    if not verify_mpin(payload.mpin, user.mpin_hash):
        next_attempt_count = (user.failed_mpin_attempts or 0) + 1
        lock_until = None
        if next_attempt_count >= settings.MPIN_LOCKOUT_FAILURES:
            lock_until = _now_utc() + timedelta(minutes=settings.MPIN_LOCKOUT_MINUTES)
        await repo.increment_failed_mpin_attempts(user, lock_until)
        await db.commit()
        if lock_until is not None:
            raise HTTPException(status_code=423, detail="Account temporarily locked. Try again later.")
        raise HTTPException(status_code=401, detail="Invalid account number or MPIN.")

    await repo.reset_failed_mpin_attempts(user)
    await repo.update_last_login(user.id)

    is_enrolled = await _is_enrolled(db, user.id)
    if user.kyc_status == "PENDING":
        is_enrolled = False

    new_device_login = False
    ip_addr = request.client.host if request.client else "127.0.0.1"
    if payload.device_fingerprint:
        fingerprint_hash = _device_hash(payload.device_fingerprint)
        known = await repo.get_known_device(user.id, fingerprint_hash)
        new_device_login = known is None
        await repo.record_known_device(user.id, fingerprint_hash)
        if new_device_login:
            audit = get_audit_vault()
            await audit.log_event(
                db=db,
                event_type=AuditEventType.NEW_DEVICE_LOGIN,
                user_id=user.id,
                actor="CUSTOMER",
                ip_address=ip_addr,
                details=f"New device login detected. Fingerprint: {fingerprint_hash}",
                payload={"device_fingerprint_hash": fingerprint_hash},
            )

    token = create_access_token(
        user_id=user.id,
        account_number=user.account_number,
        full_name=user.full_name,
        role="CUSTOMER",
    )
    mpin_changed = _as_aware(user.mpin_last_changed_at)
    mpin_rotation_required = (
        mpin_changed is not None
        and (_now_utc() - mpin_changed).days > settings.MPIN_MAX_AGE_DAYS
    )
    await db.commit()

    return MPINLoginResponse(
        success=True,
        user_id=user.id,
        account_number=user.account_number,
        full_name=user.full_name,
        session_token=token,
        is_enrolled=is_enrolled,
        kyc_status=user.kyc_status,
        mpin_rotation_required=mpin_rotation_required,
        new_device_login=new_device_login,
    )


@router.get("/auth/me", response_model=UserPublicRead, tags=["Banking Auth"])
async def get_me(
    user_id: uuid.UUID = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
) -> UserPublicRead:
    repo = UserRepository(db)
    user = await repo.get_by_id(user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="Account not found.")
    return _public_user(user, await _is_enrolled(db, user.id))


@router.post("/auth/mpin/change", tags=["Banking Auth"])
@limiter.limit("5/minute")
async def change_mpin(
    request: Request,
    payload: MPINChangeRequest,
    user_id: uuid.UUID = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
) -> dict:
    repo = UserRepository(db)
    user = await repo.get_by_id(user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="Account not found.")
    if not verify_mpin(payload.old_mpin, user.mpin_hash):
        raise HTTPException(status_code=401, detail="Invalid MPIN.")
    await repo.update_mpin_hash(user, hash_mpin(payload.new_mpin))
    await db.commit()
    return {"success": True, "message": "MPIN changed successfully."}


@router.post("/auth/refresh", tags=["Banking Auth"])
@limiter.limit("20/minute")
async def refresh_session(
    request: Request,
    claims: dict = Depends(get_current_user_claims),
) -> dict:
    token = create_access_token(
        user_id=uuid.UUID(claims["sub"]),
        account_number=claims["account_number"],
        full_name=claims["full_name"],
        role=claims.get("role", "CUSTOMER"),
    )
    return {"success": True, "session_token": token}


@router.post("/auth/freeze", tags=["Banking Auth"])
@limiter.limit("5/minute")
async def freeze_account(
    request: Request,
    user_id: uuid.UUID = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
) -> dict:
    repo = UserRepository(db)
    user = await repo.get_by_id(user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="Account not found.")
    await repo.set_active(user, False)
    await db.commit()
    return {"success": True, "message": "Account frozen."}


@router.post("/auth/reactivate", tags=["Banking Auth"], dependencies=[Depends(allow_staff)])
@limiter.limit("20/minute")
async def reactivate_account(
    request: Request,
    payload: AccountNumberRequest,
    db: AsyncSession = Depends(get_db),
) -> dict:
    repo = UserRepository(db)
    user = await repo.get_by_account_number(payload.account_number)
    if user is None:
        raise HTTPException(status_code=404, detail="Account not found.")
    await repo.set_active(user, True)
    await db.commit()
    return {"success": True, "message": "Account reactivated."}


@router.get(
    "/users/{user_id}",
    response_model=UserDetailRead,
    status_code=status.HTTP_200_OK,
    summary="Get a user and their enrollment status",
    tags=["Users"],
)
async def get_user(
    user_id: uuid.UUID, db: AsyncSession = Depends(get_db)
) -> UserDetailRead:
    repo = UserRepository(db)
    user: User | None = await repo.get_by_id_with_voiceprint(user_id)

    if user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=UserNotFoundError(str(user_id)).message,
        )

    anchor = await AnchorEmbeddingRepository(db).get_by_user_id(user_id)
    voiceprint = user.voiceprint
    is_enrolled = (voiceprint is not None) or (anchor is not None)
    recording_count = voiceprint.recording_count if voiceprint else (anchor.recording_count if anchor else 0)
    embedding_dim = len(voiceprint.embedding) if voiceprint else (anchor.embedding_dim if anchor else None)
    updated_at = voiceprint.updated_at if voiceprint else (anchor.updated_at if anchor else None)

    return UserDetailRead(
        id=user.id,
        account_number=user.account_number,
        full_name=user.full_name,
        email=user.email,
        created_at=user.created_at,
        is_enrolled=is_enrolled,
        recording_count=recording_count,
        embedding_dimension=embedding_dim,
        voiceprint_updated_at=updated_at,
    )
