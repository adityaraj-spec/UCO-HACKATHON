"""
app/models/layer2/rolling_embedding.py

Rolling embedding pool model.

Stores up to MAX_ROLLING_EMBEDDINGS (10) recent high-confidence auth embeddings
per user. These adapt to gradual voice drift (aging, environment changes) while
the anchor always serves as the stable baseline.

Update policy (enforced in embedding_worker.py):
  1. score > HIGH_CONFIDENCE_THRESHOLD (0.92)
  2. 3 consecutive successes within 24 hours
  3. Cosine distance from anchor < DRIFT_ALERT_DISTANCE (0.15)
  4. Fraud engine clearance for new device/network
"""

import uuid
from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, Integer, LargeBinary, String, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.database.base import Base


class RollingEmbedding(Base):
    """Single entry in a user's rolling embedding pool."""

    __tablename__ = "rolling_embeddings"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), nullable=False, index=True
    )
    auth_history_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), nullable=True
    )  # FK to auth_history

    # Encrypted cancelable template (same pipeline as anchor)
    encrypted_embedding: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    encryption_nonce: Mapped[bytes] = mapped_column(LargeBinary(12), nullable=False)
    embedding_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    key_id: Mapped[str] = mapped_column(String(128), nullable=False)

    # Quality of the auth sample that produced this embedding
    auth_similarity_score: Mapped[float] = mapped_column(Float, nullable=False)
    cosine_distance_from_anchor: Mapped[float] = mapped_column(Float, nullable=True)

    # Pool management
    pool_position: Mapped[int] = mapped_column(Integer, nullable=False)  # 0-9 (LRU position)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    channel: Mapped[str] = mapped_column(String(32), nullable=False, default="MOBILE_APP")

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    deactivated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    def __repr__(self) -> str:
        return (
            f"<RollingEmbedding id={self.id} user_id={self.user_id} "
            f"pos={self.pool_position} score={self.auth_similarity_score:.3f}>"
        )
