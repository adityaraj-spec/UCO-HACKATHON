"""dynamic voice kyc challenge sessions

Revision ID: 20260722_0200
Revises: 20260722_0100
Create Date: 2026-07-22 02:00:00.000000
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "20260722_0200"
down_revision = "20260722_0100"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "kyc_enrollment_sessions",
        sa.Column("session_id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("phone_number", sa.String(length=20), nullable=False),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("status", sa.String(length=20), nullable=False, server_default="STARTED"),
        sa.Column("embedding_1", sa.JSON(), nullable=True),
        sa.Column("embedding_2", sa.JSON(), nullable=True),
        sa.Column("salt_id", sa.String(length=50), nullable=True),
        sa.Column("bio_hash", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="SET NULL"),
    )
    op.create_index("ix_kyc_enrollment_sessions_phone_number", "kyc_enrollment_sessions", ["phone_number"])

    op.create_table(
        "voice_challenges",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("session_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("phone_number", sa.String(length=20), nullable=False),
        sa.Column("attempt_no", sa.SmallInteger(), nullable=False),
        sa.Column("sentence_text", sa.Text(), nullable=False),
        sa.Column("issued_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("used", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("asr_transcript", sa.Text(), nullable=True),
        sa.Column("asr_match_score", sa.Numeric(5, 2), nullable=True),
        sa.Column("status", sa.String(length=20), nullable=False, server_default="ISSUED"),
        sa.ForeignKeyConstraint(["session_id"], ["kyc_enrollment_sessions.session_id"], ondelete="CASCADE"),
    )
    op.create_index("ix_voice_challenges_session_id", "voice_challenges", ["session_id"])


def downgrade() -> None:
    op.drop_index("ix_voice_challenges_session_id", table_name="voice_challenges")
    op.drop_table("voice_challenges")
    op.drop_index("ix_kyc_enrollment_sessions_phone_number", table_name="kyc_enrollment_sessions")
    op.drop_table("kyc_enrollment_sessions")
