"""
KYC enrollment session and dynamic voice challenge models.
"""

import uuid
from datetime import datetime, timezone

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.database.base import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class KYCEnrollmentSession(Base):
    __tablename__ = "kyc_enrollment_sessions"

    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    phone_number: Mapped[str] = mapped_column(sa.String(20), nullable=False, index=True)
    user_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), sa.ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    status: Mapped[str] = mapped_column(sa.String(20), nullable=False, default="STARTED")
    embedding_1: Mapped[list[float] | None] = mapped_column(sa.JSON, nullable=True)
    embedding_2: Mapped[list[float] | None] = mapped_column(sa.JSON, nullable=True)
    salt_id: Mapped[str | None] = mapped_column(sa.String(50), nullable=True)
    bio_hash: Mapped[str | None] = mapped_column(sa.Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True), nullable=False, default=_utcnow
    )
    completed_at: Mapped[datetime | None] = mapped_column(sa.DateTime(timezone=True), nullable=True)


class VoiceChallenge(Base):
    __tablename__ = "voice_challenges"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        sa.ForeignKey("kyc_enrollment_sessions.session_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    phone_number: Mapped[str] = mapped_column(sa.String(20), nullable=False)
    attempt_no: Mapped[int] = mapped_column(sa.SmallInteger, nullable=False)
    sentence_text: Mapped[str] = mapped_column(sa.Text, nullable=False)
    issued_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True), nullable=False, default=_utcnow
    )
    expires_at: Mapped[datetime] = mapped_column(sa.DateTime(timezone=True), nullable=False)
    used: Mapped[bool] = mapped_column(sa.Boolean, nullable=False, default=False)
    asr_transcript: Mapped[str | None] = mapped_column(sa.Text, nullable=True)
    asr_match_score: Mapped[float | None] = mapped_column(sa.Numeric(5, 2), nullable=True)
    status: Mapped[str] = mapped_column(sa.String(20), nullable=False, default="ISSUED")
