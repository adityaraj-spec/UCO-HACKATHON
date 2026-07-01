"""
app/models/layer2/enrollment.py

Enrollment session tracking model.
Tracks individual enrollment sessions with status, quality scores, and sample counts.
"""

import uuid
from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.database.base import Base


class EnrollmentSession(Base):
    """
    Tracks an enrollment session from start to completion.

    Each enrollment requires exactly MIN_ENROLLMENT_SAMPLES (5) audio samples.
    Sessions expire after ENROLLMENT_SESSION_TTL_SECONDS (600 seconds).
    Raw audio is deleted immediately after embedding extraction.
    """

    __tablename__ = "enrollment_sessions"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), nullable=False, index=True
    )
    session_token: Mapped[str] = mapped_column(String(512), nullable=False, unique=True)
    consent_token_used: Mapped[str] = mapped_column(String(512), nullable=True)

    status: Mapped[str] = mapped_column(
        String(32), nullable=False, default="PENDING"
    )  # PENDING | IN_PROGRESS | COMPLETED | FAILED | EXPIRED

    samples_required: Mapped[int] = mapped_column(Integer, nullable=False, default=5)
    samples_received: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    samples_accepted: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    # Quality metrics
    avg_quality_score: Mapped[float] = mapped_column(Float, nullable=True)
    avg_snr_db: Mapped[float] = mapped_column(Float, nullable=True)
    enrollment_quality: Mapped[str] = mapped_column(
        String(16), nullable=True
    )  # LOW | MEDIUM | HIGH

    # Channel and metadata
    channel: Mapped[str] = mapped_column(String(32), nullable=False, default="MOBILE_APP")
    ip_address: Mapped[str] = mapped_column(String(64), nullable=True)
    device_fingerprint: Mapped[str] = mapped_column(String(256), nullable=True)

    error_message: Mapped[str] = mapped_column(Text, nullable=True)
    completed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=True)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )

    def __repr__(self) -> str:
        return (
            f"<EnrollmentSession id={self.id} user_id={self.user_id} "
            f"status={self.status} samples={self.samples_accepted}/{self.samples_required}>"
        )
