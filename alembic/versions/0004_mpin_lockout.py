"""mpin lockout and known device tracking

Revision ID: 0004_mpin_lockout
Revises: 0003_banking_identity
Create Date: 2026-07-04 00:00:00
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "0004_mpin_lockout"
down_revision: Union[str, None] = "0003_banking_identity"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "users",
        sa.Column("failed_mpin_attempts", sa.Integer(), nullable=False, server_default="0"),
    )
    op.add_column("users", sa.Column("mpin_locked_until", sa.DateTime(timezone=True), nullable=True))
    op.add_column("users", sa.Column("mpin_last_changed_at", sa.DateTime(timezone=True), nullable=True))

    op.create_table(
        "known_devices",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("device_fingerprint_hash", sa.String(length=64), nullable=False),
        sa.Column("label", sa.String(length=128), nullable=True),
        sa.Column("first_seen_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("last_seen_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
    )
    op.create_index("ix_known_devices_user_id", "known_devices", ["user_id"])
    op.create_index("ix_known_devices_device_fingerprint_hash", "known_devices", ["device_fingerprint_hash"])


def downgrade() -> None:
    op.drop_index("ix_known_devices_device_fingerprint_hash", table_name="known_devices")
    op.drop_index("ix_known_devices_user_id", table_name="known_devices")
    op.drop_table("known_devices")
    op.drop_column("users", "mpin_last_changed_at")
    op.drop_column("users", "mpin_locked_until")
    op.drop_column("users", "failed_mpin_attempts")
