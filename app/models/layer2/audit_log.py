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
from typing import Any

from sqlalchemy import DateTime, String, Text, func
from sqlalchemy.dialects.postgresql import UUID, JSONB
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

    # Actor (who triggered the event)
    actor: Mapped[str] = mapped_column(String(64), nullable=False)

    # Network context
    ip_address: Mapped[str] = mapped_column(String(45), nullable=False)

    # Details
    details: Mapped[str] = mapped_column(String(512), nullable=False)

    # SHA-256 signature & previous entry link
    hash_signature: Mapped[str] = mapped_column(String(64), nullable=False)
    previous_hash: Mapped[str] = mapped_column(String(64), nullable=False)

    # Payload (stored as JSONB)
    payload: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False, index=True
    )

    def __init__(self, **kwargs: Any) -> None:
        # Safely capture payload_hash if passed, mapping it or ignoring it
        kwargs.pop("payload_hash", None)
        super().__init__(**kwargs)

    @property
    def payload_hash(self) -> str:
        import json
        import hashlib
        payload_data = self.payload or {}
        payload_str = json.dumps(payload_data, sort_keys=True)
        return hashlib.sha256(payload_str.encode()).hexdigest()

    def __repr__(self) -> str:
        return (
            f"<AuditLog id={self.id} event={self.event_type} "
            f"user={self.user_id} actor={self.actor}>"
        )
