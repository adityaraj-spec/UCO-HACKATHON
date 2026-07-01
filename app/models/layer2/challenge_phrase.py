"""
app/models/layer2/challenge_phrase.py

Challenge phrase bank for liveness verification.

Random challenge phrases prevent replay attacks by requiring the user to
speak a dynamically selected phrase. Phrases are pre-seeded and versioned.
"""

import uuid
from datetime import datetime

from sqlalchemy import Boolean, DateTime, Integer, String, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.database.base import Base


class ChallengePhrase(Base):
    """
    A single challenge phrase from the banking phrase bank.

    Phrases should be:
    - Short (4-8 words), phonemically diverse
    - Contain numbers / dates for higher entropy
    - Tested for clarity across Indian English accents
    """

    __tablename__ = "challenge_phrases"

    id: Mapped[str] = mapped_column(
        String(16), primary_key=True
    )  # e.g. "PHRASE_001"
    phrase_text: Mapped[str] = mapped_column(String(256), nullable=False)
    language: Mapped[str] = mapped_column(String(8), nullable=False, default="en-IN")

    # Phoneme diversity score (higher = better for liveness testing)
    phoneme_score: Mapped[float] = mapped_column(nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    usage_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    def __repr__(self) -> str:
        return f"<ChallengePhrase id={self.id} lang={self.language}>"


class IssuedChallenge(Base):
    """
    A time-limited challenge issued to a user for a specific auth attempt.
    Expires after CHALLENGE_TTL_SECONDS (180 seconds).
    """

    __tablename__ = "issued_challenges"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False, index=True)
    session_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    phrase_id: Mapped[str] = mapped_column(String(16), nullable=False)
    phrase_text: Mapped[str] = mapped_column(String(256), nullable=False)

    # Challenge token (JWT jti for replay protection)
    challenge_jti: Mapped[str] = mapped_column(String(128), nullable=False, unique=True)

    is_used: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    used_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    def __repr__(self) -> str:
        return (
            f"<IssuedChallenge id={self.id} phrase={self.phrase_id} "
            f"used={self.is_used}>"
        )
