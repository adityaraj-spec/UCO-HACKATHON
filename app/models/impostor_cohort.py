"""
app/models/impostor_cohort.py

SQLAlchemy model for storing dynamic impostor cohort voiceprint vectors
used by Adaptive Score Normalisation (s-norm).
"""

import uuid
from datetime import datetime, timezone

import sqlalchemy
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.database.base import Base


class ImpostorCohort(Base):
    """Stores non-matching impostor embeddings for s-norm score normalisation."""

    __tablename__ = "impostor_cohort"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )

    embedding: Mapped[list[float]] = mapped_column(
        JSONB,
        nullable=False,
    )

    speaker_label: Mapped[str] = mapped_column(
        sqlalchemy.String(100), nullable=True
    )

    created_at: Mapped[datetime] = mapped_column(
        sqlalchemy.DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
