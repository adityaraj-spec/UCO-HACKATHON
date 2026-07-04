"""
app/repositories/user_repository.py

Data-access layer for the `users` table.
"""

import uuid
from datetime import date, datetime

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models.user import KnownDevice, User


class UserRepository:
    """Encapsulates all database operations for the User model."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(self, name: str, email: str) -> User:
        """Insert a new user row and flush so its generated id is available."""
        legacy_suffix = uuid.uuid4().hex[:10]
        user = User(
            account_number=f"LEGACY-{legacy_suffix}",
            mpin_hash="legacy-dev-account",
            full_name=name,
            email=email,
            phone_number=f"000000{legacy_suffix[:4]}",
            account_type="SAVINGS",
            kyc_status="PENDING",
        )
        self.session.add(user)
        await self.session.flush()
        await self.session.refresh(user)
        return user

    async def create_bank_customer(
        self,
        account_number: str,
        mpin_hash: str,
        full_name: str,
        phone_number: str,
        dob: date,
        branch_code: str,
        account_type: str,
        email: str | None = None,
        kyc_status: str = "PENDING",
    ) -> User:
        """Create a production-shaped banking customer."""
        user = User(
            account_number=account_number,
            mpin_hash=mpin_hash,
            full_name=full_name,
            phone_number=phone_number,
            date_of_birth=dob,
            branch_code=branch_code,
            account_type=account_type,
            kyc_status=kyc_status,
            email=email,
            is_active=True,
            mpin_last_changed_at=datetime.utcnow(),
        )
        self.session.add(user)
        await self.session.flush()
        await self.session.refresh(user)
        return user

    async def get_by_id(self, user_id: uuid.UUID) -> User | None:
        """Fetch a user by primary key, or None if not found."""
        result = await self.session.execute(
            select(User).where(User.id == user_id)
        )
        return result.scalar_one_or_none()

    async def get_by_id_with_voiceprint(self, user_id: uuid.UUID) -> User | None:
        """Fetch a user eagerly loading its voiceprint relationship."""
        result = await self.session.execute(
            select(User)
            .options(selectinload(User.voiceprint))
            .where(User.id == user_id)
        )
        return result.scalar_one_or_none()

    async def get_by_email(self, email: str) -> User | None:
        """Fetch a user by unique email address, or None if not found."""
        result = await self.session.execute(
            select(User).where(User.email == email)
        )
        return result.scalar_one_or_none()

    async def get_by_account_number(self, account_number: str) -> User | None:
        """Fetch a user by bank-visible account number."""
        result = await self.session.execute(
            select(User).where(User.account_number == account_number)
        )
        return result.scalar_one_or_none()

    async def get_by_phone(self, phone: str) -> User | None:
        """Fetch a user by registered phone number."""
        result = await self.session.execute(
            select(User).where(User.phone_number == phone)
        )
        return result.scalar_one_or_none()

    async def update_last_login(self, user_id: uuid.UUID) -> None:
        await self.session.execute(
            update(User).where(User.id == user_id).values(last_login_at=datetime.utcnow())
        )

    async def increment_failed_mpin_attempts(
        self, user: User, locked_until: datetime | None
    ) -> None:
        user.failed_mpin_attempts = (user.failed_mpin_attempts or 0) + 1
        if locked_until is not None:
            user.mpin_locked_until = locked_until
        self.session.add(user)
        await self.session.flush()

    async def reset_failed_mpin_attempts(self, user: User) -> None:
        user.failed_mpin_attempts = 0
        user.mpin_locked_until = None
        self.session.add(user)
        await self.session.flush()

    async def update_mpin_hash(self, user: User, mpin_hash: str) -> None:
        user.mpin_hash = mpin_hash
        user.mpin_last_changed_at = datetime.utcnow()
        user.failed_mpin_attempts = 0
        user.mpin_locked_until = None
        self.session.add(user)
        await self.session.flush()

    async def set_active(self, user: User, is_active: bool) -> None:
        user.is_active = is_active
        self.session.add(user)
        await self.session.flush()

    async def get_known_device(
        self, user_id: uuid.UUID, fingerprint_hash: str
    ) -> KnownDevice | None:
        result = await self.session.execute(
            select(KnownDevice).where(
                KnownDevice.user_id == user_id,
                KnownDevice.device_fingerprint_hash == fingerprint_hash,
            )
        )
        return result.scalar_one_or_none()

    async def record_known_device(
        self,
        user_id: uuid.UUID,
        fingerprint_hash: str,
        label: str | None = None,
    ) -> KnownDevice:
        device = await self.get_known_device(user_id, fingerprint_hash)
        if device is None:
            device = KnownDevice(
                user_id=user_id,
                device_fingerprint_hash=fingerprint_hash,
                label=label,
            )
            self.session.add(device)
        else:
            device.last_seen_at = datetime.utcnow()
            self.session.add(device)
        await self.session.flush()
        await self.session.refresh(device)
        return device

    async def exists(self, user_id: uuid.UUID) -> bool:
        """Return True if a user with the given id exists."""
        result = await self.session.execute(
            select(User.id).where(User.id == user_id)
        )
        return result.scalar_one_or_none() is not None
