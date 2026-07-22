"""create legal_guardian_requests table

Revision ID: 0005_legal_guardian
Revises: 0004_mpin_lockout
Create Date: 2026-07-04 12:00:00.000000
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "0005_legal_guardian"
down_revision: Union[str, None] = "0004_mpin_lockout"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "legal_guardian_requests",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("pdf_path", sa.String(length=512), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False, server_default="PENDING_REVIEW"),
        sa.Column("branch_manager_approved", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("branch_manager_id", sa.String(length=128), nullable=True),
        sa.Column("compliance_officer_approved", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("compliance_officer_id", sa.String(length=128), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_legal_guardian_requests_user_id", "legal_guardian_requests", ["user_id"])


def downgrade() -> None:
    op.drop_index("ix_legal_guardian_requests_user_id", table_name="legal_guardian_requests")
    op.drop_table("legal_guardian_requests")
