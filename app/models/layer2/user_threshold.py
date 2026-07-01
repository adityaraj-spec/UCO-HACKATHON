"""
app/models/layer2/user_threshold.py

Per-user adaptive threshold state for authentication decisions.

The adaptive threshold engine adjusts the cosine similarity pass threshold
per user based on their authentication history, enrollment quality, and
environmental conditions (illness windows, consecutive failures, etc.).
"""

import uuid
from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, Integer, String, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.database.base import Base


class UserThreshold(Base):
    """
    Stores the current adaptive threshold state for a user.

    threshold_current = effective threshold used in authentication decisions.
    It starts at the global baseline and adjusts based on:
    - Consecutive auth failures (lower by 0.02 each time, floor: baseline-0.10)
    - High-SNR environment (raise by 0.01 after sustained good quality auths)
    - Illness window: temporary relaxation when FRR spikes detected
    - Fraud engine high-risk: raise by 0.05 until clearance
    """

    __tablename__ = "user_thresholds"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), nullable=False, unique=True, index=True
    )

    # Threshold values
    threshold_baseline: Mapped[float] = mapped_column(Float, nullable=False, default=0.65)
    threshold_current: Mapped[float] = mapped_column(Float, nullable=False, default=0.65)
    threshold_floor: Mapped[float] = mapped_column(Float, nullable=False, default=0.50)
    threshold_ceiling: Mapped[float] = mapped_column(Float, nullable=False, default=0.85)

    # Illness tolerance window
    illness_window_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    illness_window_expires_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    illness_window_relaxation: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)

    # Failure tracking (for threshold adjustment)
    consecutive_failures: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    failures_last_24h: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    failures_window_reset_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # Risk elevation
    fraud_risk_elevated: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    fraud_risk_elevation: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)

    # Calibration state
    enrollment_quality_score: Mapped[float] = mapped_column(Float, nullable=True)
    last_calibration_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=True)
    last_successful_auth_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # Adjustment history (last adjustment reason for audit)
    last_adjustment_reason: Mapped[str] = mapped_column(String(128), nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )

    def __repr__(self) -> str:
        return (
            f"<UserThreshold user_id={self.user_id} "
            f"current={self.threshold_current:.3f} baseline={self.threshold_baseline:.3f}>"
        )
