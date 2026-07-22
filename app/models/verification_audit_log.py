"""
app/models/verification_audit_log.py

SQLAlchemy model for immutable, append-only verification audit logs required for data governance.
"""

import uuid
from datetime import datetime, timezone

import sqlalchemy
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.database.base import Base


class VerificationAuditLog(Base):
    """Immutable audit trail for every verification attempt."""

    __tablename__ = "verification_audit_log"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )

    user_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), nullable=True, index=True
    )

    decision: Mapped[str] = mapped_column(
        sqlalchemy.String(50), nullable=False
    )

    raw_similarity: Mapped[float] = mapped_column(
        sqlalchemy.Float, nullable=False
    )

    snorm_score: Mapped[float | None] = mapped_column(
        sqlalchemy.Float, nullable=True
    )

    model_version: Mapped[str] = mapped_column(
        sqlalchemy.String(50), nullable=False
    )

    request_id_hash: Mapped[str] = mapped_column(
        sqlalchemy.String(64), nullable=False
    )

    created_at: Mapped[datetime] = mapped_column(
        sqlalchemy.DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        index=True,
    )
