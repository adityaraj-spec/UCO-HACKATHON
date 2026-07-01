"""
app/models/layer2/audit_log.py

Immutable, tamper-evident audit vault.

Each audit log entry is chained to the previous one via a SHA-256 hash
of the previous entry, making any retroactive modification detectable.
This satisfies:
  - PMLA 2002: 5-year audit retention
  - RBI Digital Payment Security Controls: complete transaction audit trail
  - DPDP Act 2023: consent and biometric processing audit
"""

import uuid
from datetime import datetime

from sqlalchemy import DateTime, String, Text, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.database.base import Base


class AuditLog(Base):
    """
    Immutable audit log entry.

    Rules:
    - NEVER update or delete rows in this table.
    - Each entry stores a SHA-256 of the previous entry (chaining).
    - payload_hash is SHA-256 of the JSON-serialized event payload.
    - The combination (prev_hash, payload_hash) makes tampering detectable.
    """

    __tablename__ = "audit_logs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), nullable=True, index=True
    )

    # Event classification
    event_type: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    # Sub-type for filtering (e.g. "MOBILE_APP", "IVR")
    event_source: Mapped[str] = mapped_column(String(64), nullable=True)

    # Actor (who triggered the event)
    actor_id: Mapped[str] = mapped_column(String(128), nullable=True)
    actor_role: Mapped[str] = mapped_column(String(32), nullable=True)

    # Network context
    ip_address: Mapped[str] = mapped_column(String(64), nullable=True)
    device_fingerprint: Mapped[str] = mapped_column(String(256), nullable=True)

    # Payload (JSON-serialized event data — no raw biometric data)
    event_payload: Mapped[str] = mapped_column(Text, nullable=True)
    # SHA-256 of event_payload (integrity check)
    payload_hash: Mapped[str] = mapped_column(String(64), nullable=False)

    # Chain integrity — SHA-256 of the previous audit entry
    prev_entry_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), nullable=True
    )
    prev_hash: Mapped[str] = mapped_column(String(64), nullable=True)

    # Status / outcome
    outcome: Mapped[str] = mapped_column(
        String(32), nullable=False, default="SUCCESS"
    )  # SUCCESS | FAILURE | WARNING

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False, index=True
    )

    def __repr__(self) -> str:
        return (
            f"<AuditLog id={self.id} event={self.event_type} "
            f"user={self.user_id} outcome={self.outcome}>"
        )
