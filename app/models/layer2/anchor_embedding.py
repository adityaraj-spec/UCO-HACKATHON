"""
app/models/layer2/anchor_embedding.py

Anchor embedding storage model.

The anchor embedding is the stable reference created at enrollment time
(average of 5+ high-quality samples). It is:
 - Transformed via BioHash (cancelable biometrics)
 - Encrypted with AES-256-GCM using an HSM-backed key
 - NEVER stored as a raw voiceprint

The pgvector column stores the ENCRYPTED embedding bytes as a recognizable
float vector for pgvector-based similarity search (only available after
decryption + inverse BioHash — which requires the user-specific key).
"""

import uuid
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import Boolean, DateTime, Float, Integer, LargeBinary, String, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.base import Base


class AnchorEmbedding(Base):
    """
    The permanent, never-overwritten anchor voiceprint for a user.

    Stores:
    - encrypted_embedding: AES-256-GCM ciphertext of the BioHash-transformed embedding
    - embedding_hash:      SHA-256 of the plaintext embedding (for audit/integrity checks)
    - key_id:              HSM key ID used for encryption
    - template_version:    increments on key rotation (revocation & re-enrollment)
    - dim:                 embedding dimension (192 for ECAPA-TDNN)
    """

    __tablename__ = "anchor_embeddings"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), nullable=False, unique=True, index=True
    )

    # Encrypted cancelable template (ciphertext blob)
    encrypted_embedding: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    # Nonce / IV used for AES-GCM (stored alongside ciphertext)
    encryption_nonce: Mapped[bytes] = mapped_column(LargeBinary(12), nullable=False)
    # SHA-256 hex of the plaintext embedding for integrity auditing
    embedding_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    # HSM key ID that encrypted this embedding
    key_id: Mapped[str] = mapped_column(String(128), nullable=False)

    # Template lifecycle
    template_version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)

    # Enrollment quality metadata
    enrollment_quality_score: Mapped[float] = mapped_column(Float, nullable=True)
    recording_count: Mapped[int] = mapped_column(Integer, nullable=False, default=5)
    embedding_dim: Mapped[int] = mapped_column(Integer, nullable=False, default=192)

    # Enrollment channel
    enrolled_via: Mapped[str] = mapped_column(String(32), nullable=False, default="MOBILE_APP")

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )
    deactivated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=True)

    def __repr__(self) -> str:
        return (
            f"<AnchorEmbedding id={self.id} user_id={self.user_id} "
            f"version={self.template_version} active={self.is_active}>"
        )
