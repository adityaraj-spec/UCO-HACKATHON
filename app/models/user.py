"""
app/models/user.py

SQLAlchemy ORM model for the `users` table.

Represents an enrolled customer / bank account holder whose voice will be
verified against incoming calls.
"""

import uuid
from datetime import date, datetime
from enum import Enum

from sqlalchemy import Boolean, Date, DateTime, ForeignKey, Integer, String, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.base import Base


class User(Base):
    """A bank customer enrolled in PhaseGuard's voice verification system."""

    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        nullable=False,
    )
    account_number: Mapped[str] = mapped_column(
        String(20), nullable=False, unique=True, index=True
    )
    mpin_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    full_name: Mapped[str] = mapped_column(String(255), nullable=False)
    phone_number: Mapped[str] = mapped_column(
        String(20), nullable=False, unique=True, index=True
    )
    date_of_birth: Mapped[date | None] = mapped_column(Date, nullable=True)
    branch_code: Mapped[str | None] = mapped_column(String(10), nullable=True)
    account_type: Mapped[str] = mapped_column(String(16), nullable=False, default="SAVINGS")
    kyc_status: Mapped[str] = mapped_column(String(16), nullable=False, default="PENDING")
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    last_login_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    failed_mpin_attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    mpin_locked_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    mpin_last_changed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    email: Mapped[str | None] = mapped_column(
        String(255), nullable=True, unique=True, index=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # Relationships
    voiceprint: Mapped["Voiceprint"] = relationship(  # noqa: F821
        "Voiceprint",
        back_populates="user",
        uselist=False,
        cascade="all, delete-orphan",
    )
    verification_logs: Mapped[list["VerificationLog"]] = relationship(  # noqa: F821
        "VerificationLog",
        back_populates="user",
        cascade="all, delete-orphan",
    )
    risk_logs: Mapped[list["RiskLog"]] = relationship(  # noqa: F821
        "RiskLog",
        back_populates="user",
        cascade="all, delete-orphan",
    )
    known_devices: Mapped[list["KnownDevice"]] = relationship(
        "KnownDevice",
        back_populates="user",
        cascade="all, delete-orphan",
    )

    @property
    def name(self) -> str:
        """Backward-compatible alias for older service code/tests."""
        return self.full_name

    @name.setter
    def name(self, value: str) -> None:
        self.full_name = value

    def __repr__(self) -> str:
        return f"<User id={self.id} account_number={self.account_number!r}>"


class AccountType(str, Enum):
    SAVINGS = "SAVINGS"
    CURRENT = "CURRENT"
    NRE = "NRE"


class KYCStatus(str, Enum):
    PENDING = "PENDING"
    VERIFIED = "VERIFIED"
    REJECTED = "REJECTED"


class KnownDevice(Base):
    """Hashed device fingerprints seen during MPIN login."""

    __tablename__ = "known_devices"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    device_fingerprint_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    label: Mapped[str | None] = mapped_column(String(128), nullable=True)
    first_seen_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    last_seen_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )

    user: Mapped[User] = relationship("User", back_populates="known_devices")
