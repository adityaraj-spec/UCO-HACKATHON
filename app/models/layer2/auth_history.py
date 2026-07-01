"""
app/models/layer2/auth_history.py

Completed authentication result log.

Every authentication attempt (pass/fail/step-up) is recorded here.
This table supports: audit queries, rolling embedding update eligibility
checks (3 consecutive successes in 24h), adaptive threshold calibration,
and fraud pattern analysis.
"""

import uuid
from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.database.base import Base


class AuthHistory(Base):
    """Immutable record of each authentication attempt."""

    __tablename__ = "auth_history"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), nullable=False, index=True
    )
    session_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    challenge_phrase_id: Mapped[str] = mapped_column(String(64), nullable=True)

    # Auth outcome
    auth_result: Mapped[str] = mapped_column(
        String(32), nullable=False
    )  # PASS | FAIL | STEP_UP_REQUIRED | SPOOF_DETECTED | ERROR

    # Scores
    anchor_similarity_score: Mapped[float] = mapped_column(Float, nullable=True)
    rolling_avg_score: Mapped[float] = mapped_column(Float, nullable=True)
    weighted_score: Mapped[float] = mapped_column(Float, nullable=True)  # Combined
    threshold_used: Mapped[float] = mapped_column(Float, nullable=True)
    fraud_risk_score: Mapped[float] = mapped_column(Float, nullable=True)

    # Anti-spoofing
    antispoof_result: Mapped[str] = mapped_column(String(32), nullable=True)  # PASS | FAIL
    antispoof_confidence: Mapped[float] = mapped_column(Float, nullable=True)
    liveness_verified: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    # Context
    channel: Mapped[str] = mapped_column(String(32), nullable=False, default="MOBILE_APP")
    ip_address: Mapped[str] = mapped_column(String(64), nullable=True)
    device_fingerprint: Mapped[str] = mapped_column(String(256), nullable=True)
    factor1_present: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    # Risk
    risk_level: Mapped[str] = mapped_column(
        String(16), nullable=False, default="UNKNOWN"
    )  # LOW | MEDIUM | HIGH | CRITICAL

    # Rolling update eligibility — set True when score qualifies
    rolling_update_queued: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    rolling_update_applied: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    reason_code: Mapped[str] = mapped_column(String(64), nullable=True)
    error_message: Mapped[str] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False, index=True
    )

    def __repr__(self) -> str:
        return (
            f"<AuthHistory id={self.id} user_id={self.user_id} "
            f"result={self.auth_result} score={self.weighted_score}>"
        )
