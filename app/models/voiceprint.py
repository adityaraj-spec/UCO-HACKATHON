"""
app/models/voiceprint.py

SQLAlchemy ORM model for the `voiceprints` table.

Stores the averaged ECAPA-TDNN speaker embedding (192-dimensional vector)
for an enrolled user, persisted via pgvector's VECTOR column type so that
similarity search can be performed directly inside PostgreSQL if desired.
"""

import uuid
from datetime import datetime

from pgvector.sqlalchemy import Vector
import sqlalchemy
from sqlalchemy import DateTime, ForeignKey, Integer, LargeBinary, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.config import get_settings
from app.database.base import Base

settings = get_settings()


class Voiceprint(Base):
    """
    A single enrolled voiceprint (averaged speaker embedding) for a user.

    Each user has exactly one active voiceprint, created/updated by the
    enrollment workflow (averaging embeddings from 30-50 utterances).
    """

    __tablename__ = "voiceprints"

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
        unique=True,
        index=True,
    )

    # Legacy pgvector column. New writes keep this NULL and store encrypted
    # material in the AES-GCM fields below.
    _raw_embedding_column: Mapped[list[float] | None] = mapped_column(
        "embedding", Vector(settings.EMBEDDING_DIM), nullable=True
    )

    encrypted_embedding: Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True)
    embedding_nonce: Mapped[bytes | None] = mapped_column(LargeBinary(12), nullable=True)
    embedding_key_id: Mapped[str | None] = mapped_column(sqlalchemy.String(64), nullable=True)

    # Legacy BioHash column. New writes keep this NULL and encrypt BioHash
    # defense-in-depth in the fields below.
    _raw_biohash_column: Mapped[str | None] = mapped_column(
        "biohash", sqlalchemy.String(256), nullable=True
    )
    encrypted_biohash: Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True)
    biohash_nonce: Mapped[bytes | None] = mapped_column(LargeBinary(12), nullable=True)
    biohash_key_id: Mapped[str | None] = mapped_column(sqlalchemy.String(64), nullable=True)

    model_version: Mapped[str] = mapped_column(
        sqlalchemy.String(50), nullable=False, server_default="ecapa-voxceleb"
    )

    needs_re_enrollment: Mapped[bool] = mapped_column(
        sqlalchemy.Boolean, nullable=False, default=False
    )

    recording_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="voiceprint")  # noqa: F821

    def __repr__(self) -> str:
        return (
            f"<Voiceprint id={self.id} user_id={self.user_id} "
            f"recording_count={self.recording_count}>"
        )

    @property
    def embedding(self) -> list[float] | None:
        return getattr(self, "_plaintext_embedding", None) or self._raw_embedding_column

    @embedding.setter
    def embedding(self, value: list[float] | None) -> None:
        self._plaintext_embedding = value

    @property
    def biohash(self) -> str | None:
        return getattr(self, "_plaintext_biohash", None) or self._raw_biohash_column

    @biohash.setter
    def biohash(self, value: str | None) -> None:
        self._plaintext_biohash = value

    def attach_plaintext(self, embedding: list[float] | None, biohash: str | None = None) -> None:
        """Attach decrypted values for in-process use without marking DB columns raw."""
        self._plaintext_embedding = embedding
        self._plaintext_biohash = biohash
