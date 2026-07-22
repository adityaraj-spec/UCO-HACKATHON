"""
app/models/verification_log.py

SQLAlchemy ORM model for the `verification_logs` table.

Each row records the outcome of a single Layer 2 (speaker verification)
attempt: the raw Layer 2 similarity score and the resulting identity decision.
"""

import uuid
from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, String, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.base import Base


class VerificationLog(Base):
    """A historical record of one speaker-verification attempt."""

    __tablename__ = "verification_logs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        nullable=False,
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    similarity_score: Mapped[float] = mapped_column(Float, nullable=False)

    # "verified" | "step_up" | "mismatch" | "REPLAY_DETECTED"
    decision: Mapped[str] = mapped_column(String(32), nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # Relationships
    user: Mapped["User"] = relationship(  # noqa: F821
        "User", back_populates="verification_logs"
    )

    def __repr__(self) -> str:
        return (
            f"<VerificationLog id={self.id} user_id={self.user_id} "
            f"score={self.similarity_score:.4f} decision={self.decision!r}>"
        )
