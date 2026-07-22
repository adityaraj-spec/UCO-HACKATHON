"""voiceprint encryption at rest

Revision ID: 20260722_0100
Revises: 20260722_0000
Create Date: 2026-07-22 01:00:00.000000
"""

from alembic import op
import sqlalchemy as sqla


revision = "20260722_0100"
down_revision = "20260722_0000"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("voiceprints", sqla.Column("encrypted_embedding", sqla.LargeBinary(), nullable=True))
    op.add_column("voiceprints", sqla.Column("embedding_nonce", sqla.LargeBinary(length=12), nullable=True))
    op.add_column("voiceprints", sqla.Column("embedding_key_id", sqla.String(length=64), nullable=True))
    op.add_column("voiceprints", sqla.Column("encrypted_biohash", sqla.LargeBinary(), nullable=True))
    op.add_column("voiceprints", sqla.Column("biohash_nonce", sqla.LargeBinary(length=12), nullable=True))
    op.add_column("voiceprints", sqla.Column("biohash_key_id", sqla.String(length=64), nullable=True))
    op.alter_column("voiceprints", "embedding", nullable=True)


def downgrade() -> None:
    op.alter_column("voiceprints", "embedding", nullable=False)
    op.drop_column("voiceprints", "biohash_key_id")
    op.drop_column("voiceprints", "biohash_nonce")
    op.drop_column("voiceprints", "encrypted_biohash")
    op.drop_column("voiceprints", "embedding_key_id")
    op.drop_column("voiceprints", "embedding_nonce")
    op.drop_column("voiceprints", "encrypted_embedding")
