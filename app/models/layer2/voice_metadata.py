"""
app/models/layer2/voice_metadata.py

Voice enrollment quality metadata.

Stores per-sample and aggregate quality metrics from enrollment.
Does NOT store any audio data or raw embeddings.
"""

import uuid
from datetime import datetime

from sqlalchemy import DateTime, Float, Integer, String, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.database.base import Base


class VoiceMetadata(Base):
    """
    Enrollment quality metadata per user.

    Tracks aggregate quality of enrollment samples for:
    - Adaptive threshold calibration
    - Enrollment quality audit
    - Debug/support (no biometric data exposed)
    """

    __tablename__ = "voice_metadata"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), nullable=False, unique=True, index=True
    )
    enrollment_session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), nullable=True
    )

    # Aggregate quality metrics
    sample_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    avg_snr_db: Mapped[float] = mapped_column(Float, nullable=True)
    min_snr_db: Mapped[float] = mapped_column(Float, nullable=True)
    max_snr_db: Mapped[float] = mapped_column(Float, nullable=True)
    avg_voice_ratio: Mapped[float] = mapped_column(Float, nullable=True)
    avg_duration_seconds: Mapped[float] = mapped_column(Float, nullable=True)
    avg_quality_score: Mapped[float] = mapped_column(Float, nullable=True)

    # Overall enrollment quality
    enrollment_quality: Mapped[str] = mapped_column(
        String(16), nullable=False, default="UNKNOWN"
    )  # LOW | MEDIUM | HIGH | EXCELLENT

    # Channel info (relevant for codec normalization)
    enrolled_via: Mapped[str] = mapped_column(String(32), nullable=False, default="MOBILE_APP")
    sample_rate_used: Mapped[int] = mapped_column(Integer, nullable=False, default=16000)
    embedding_model: Mapped[str] = mapped_column(String(64), nullable=False, default="ECAPA-TDNN")

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )

    def __repr__(self) -> str:
        return (
            f"<VoiceMetadata user_id={self.user_id} "
            f"quality={self.enrollment_quality} snr={self.avg_snr_db:.1f}dB>"
        )
