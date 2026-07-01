"""
app/models/layer2/emergency_contact.py

Emergency access framework models.

Allows registered emergency contacts to access limited account functionality
when the account holder is incapacitated. Per RBI customer protection
guidelines, voice biometrics NEVER gates emergency access — alternative
verification paths are always available.
"""

import uuid
from datetime import datetime

from sqlalchemy import Boolean, DateTime, String, Text, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.database.base import Base


class EmergencyContact(Base):
    """
    Registered emergency contact for a user's account.

    - Max 2 contacts per account (MAX_EMERGENCY_CONTACTS)
    - Contact is verified via OTP at registration time
    - Access scope is pre-authorized by account holder
    """

    __tablename__ = "emergency_contacts"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), nullable=False, index=True
    )

    # Contact details (encrypted in production)
    contact_name: Mapped[str] = mapped_column(String(256), nullable=False)
    contact_phone_encrypted: Mapped[str] = mapped_column(String(512), nullable=False)
    contact_phone_hash: Mapped[str] = mapped_column(String(64), nullable=False)  # For lookup
    relationship_type: Mapped[str] = mapped_column(
        String(64), nullable=False
    )  # SPOUSE | PARENT | CHILD | SIBLING | GUARDIAN | OTHER

    # Access permissions (pre-authorized by account holder)
    access_scope: Mapped[str] = mapped_column(
        String(32), nullable=False, default="READ_ONLY"
    )  # READ_ONLY | LIMITED_TRANSACTION | FULL_TRANSACTION

    max_transaction_limit_inr: Mapped[int] = mapped_column(
        nullable=False, default=10000
    )  # INR limit for LIMITED_TRANSACTION

    # Verification status
    is_verified: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    verification_method: Mapped[str] = mapped_column(String(32), nullable=True)  # OTP | BRANCH
    verified_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=True)

    # Lifecycle
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    activation_window_days: Mapped[int] = mapped_column(nullable=False, default=7)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )
    deactivated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=True)

    def __repr__(self) -> str:
        return (
            f"<EmergencyContact id={self.id} user_id={self.user_id} "
            f"scope={self.access_scope} verified={self.is_verified}>"
        )


class EmergencyAccessEvent(Base):
    """
    Log of emergency access activations and expirations.
    Every activation is time-limited to 72 hours.
    """

    __tablename__ = "emergency_access_events"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    emergency_contact_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), nullable=False, index=True
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), nullable=False, index=True
    )

    # Activation details
    activation_reason: Mapped[str] = mapped_column(Text, nullable=True)
    verification_method_used: Mapped[str] = mapped_column(String(32), nullable=False)
    verifying_agent_id: Mapped[str] = mapped_column(String(128), nullable=True)

    # Access session
    access_token_jti: Mapped[str] = mapped_column(String(128), nullable=False, unique=True)
    access_scope_granted: Mapped[str] = mapped_column(String(32), nullable=False)
    ip_address: Mapped[str] = mapped_column(String(64), nullable=True)

    # Status
    status: Mapped[str] = mapped_column(
        String(32), nullable=False, default="ACTIVE"
    )  # ACTIVE | EXPIRED | REVOKED

    activated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    revoked_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=True)
    revoked_by: Mapped[str] = mapped_column(String(128), nullable=True)

    # Notification status
    account_holder_notified: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    notified_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=True)

    def __repr__(self) -> str:
        return (
            f"<EmergencyAccessEvent id={self.id} user_id={self.user_id} "
            f"status={self.status} scope={self.access_scope_granted}>"
        )
