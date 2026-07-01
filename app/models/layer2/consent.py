"""
app/models/layer2/consent.py

DPDP Act 2023 compliant consent management models.

Biometric data is classified as Sensitive Personal Data under DPDP Act 2023.
Requires explicit, granular, informed consent — separate from general T&C.
"""

import uuid
from datetime import datetime

from sqlalchemy import Boolean, DateTime, String, Text, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.database.base import Base


class Consent(Base):
    """
    Current consent state for a user's biometric data processing.

    - A user MUST have an active consent record before enrollment.
    - Withdrawing consent triggers automatic deletion within 7 days.
    - One active consent per user at any time.
    """

    __tablename__ = "consents"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), nullable=False, unique=True, index=True
    )

    status: Mapped[str] = mapped_column(
        String(32), nullable=False, default="PENDING"
    )  # PENDING | ACTIVE | WITHDRAWN | EXPIRED | REVOKED

    # Consent token (signed JWT) issued at grant time
    consent_token_jti: Mapped[str] = mapped_column(String(128), nullable=False, unique=True)

    # Legal basis and purpose
    purpose: Mapped[str] = mapped_column(
        String(256), nullable=False,
        default="Voice biometric authentication for banking transaction security"
    )
    legal_basis: Mapped[str] = mapped_column(
        String(64), nullable=False, default="EXPLICIT_CONSENT"
    )

    # Opt-in confirmation
    explicit_opt_in: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    right_to_withdraw_acknowledged: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False
    )
    data_retention_acknowledged: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False
    )

    # Context
    ip_address: Mapped[str] = mapped_column(String(64), nullable=True)
    user_agent: Mapped[str] = mapped_column(String(512), nullable=True)
    channel: Mapped[str] = mapped_column(String(32), nullable=False, default="MOBILE_APP")

    # Timestamps
    granted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=True)
    withdrawn_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=True)
    deletion_scheduled_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=True)
    deletion_completed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=True)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=True)

    withdrawal_reason: Mapped[str] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )

    def __repr__(self) -> str:
        return f"<Consent id={self.id} user_id={self.user_id} status={self.status}>"


class ConsentHistory(Base):
    """
    Audit trail of all consent state changes.
    Every grant, withdrawal, and expiry is recorded here.
    Must be retained for 5 years per PMLA requirements.
    """

    __tablename__ = "consent_history"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    consent_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False, index=True)
    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False, index=True)

    event_type: Mapped[str] = mapped_column(
        String(32), nullable=False
    )  # CONSENT_GRANTED | CONSENT_WITHDRAWN | CONSENT_EXPIRED | DELETION_COMPLETED

    previous_status: Mapped[str] = mapped_column(String(32), nullable=True)
    new_status: Mapped[str] = mapped_column(String(32), nullable=False)

    actor: Mapped[str] = mapped_column(String(128), nullable=True)  # user_id or system
    ip_address: Mapped[str] = mapped_column(String(64), nullable=True)
    reason: Mapped[str] = mapped_column(Text, nullable=True)

    event_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    def __repr__(self) -> str:
        return (
            f"<ConsentHistory id={self.id} user_id={self.user_id} "
            f"event={self.event_type}>"
        )
