"""banking identity fields with legacy dev-row backfill

Revision ID: 0003_banking_identity
Revises: 0002_layer2_schema
Create Date: 2026-07-04 00:00:00

Existing dev seed rows are backfilled with intentionally fake identities:
account_number = LEGACY-<uuid_prefix>, phone_number = 0000000000+n, and
kyc_status = PENDING. This keeps migrations runnable without silently making
old name/email-only rows look like production bank customers.
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "0003_banking_identity"
down_revision: Union[str, None] = "0002_layer2_schema"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.alter_column("users", "name", new_column_name="full_name")
    op.alter_column("users", "email", existing_type=sa.String(length=255), nullable=True)

    op.add_column("users", sa.Column("account_number", sa.String(length=20), nullable=True))
    op.add_column("users", sa.Column("mpin_hash", sa.String(length=128), nullable=True))
    op.add_column("users", sa.Column("phone_number", sa.String(length=20), nullable=True))
    op.add_column("users", sa.Column("date_of_birth", sa.Date(), nullable=True))
    op.add_column("users", sa.Column("branch_code", sa.String(length=10), nullable=True))
    op.add_column("users", sa.Column("account_type", sa.String(length=16), nullable=True))
    op.add_column("users", sa.Column("kyc_status", sa.String(length=16), nullable=True))
    op.add_column("users", sa.Column("is_active", sa.Boolean(), nullable=True))
    op.add_column("users", sa.Column("last_login_at", sa.DateTime(timezone=True), nullable=True))

    op.execute(
        """
        WITH numbered AS (
            SELECT id, row_number() OVER (ORDER BY created_at, id) AS rn
            FROM users
        )
        UPDATE users u
        SET
            account_number = COALESCE(u.account_number, 'LEGACY-' || left(u.id::text, 8)),
            mpin_hash = COALESCE(u.mpin_hash, 'legacy-dev-account'),
            phone_number = COALESCE(u.phone_number, '0000000000' || numbered.rn::text),
            account_type = COALESCE(u.account_type, 'SAVINGS'),
            kyc_status = COALESCE(u.kyc_status, 'PENDING'),
            is_active = COALESCE(u.is_active, true)
        FROM numbered
        WHERE u.id = numbered.id
        """
    )

    op.alter_column("users", "account_number", existing_type=sa.String(length=20), nullable=False)
    op.alter_column("users", "mpin_hash", existing_type=sa.String(length=128), nullable=False)
    op.alter_column("users", "full_name", existing_type=sa.String(length=255), nullable=False)
    op.alter_column("users", "phone_number", existing_type=sa.String(length=20), nullable=False)
    op.alter_column("users", "account_type", existing_type=sa.String(length=16), nullable=False)
    op.alter_column("users", "kyc_status", existing_type=sa.String(length=16), nullable=False)
    op.alter_column("users", "is_active", existing_type=sa.Boolean(), nullable=False)

    op.create_index("ix_users_account_number", "users", ["account_number"], unique=True)
    op.create_index("ix_users_phone_number", "users", ["phone_number"], unique=True)


def downgrade() -> None:
    op.drop_index("ix_users_phone_number", table_name="users")
    op.drop_index("ix_users_account_number", table_name="users")
    op.drop_column("users", "last_login_at")
    op.drop_column("users", "is_active")
    op.drop_column("users", "kyc_status")
    op.drop_column("users", "account_type")
    op.drop_column("users", "branch_code")
    op.drop_column("users", "date_of_birth")
    op.drop_column("users", "phone_number")
    op.drop_column("users", "mpin_hash")
    op.drop_column("users", "account_number")
    op.alter_column("users", "email", existing_type=sa.String(length=255), nullable=False)
    op.alter_column("users", "full_name", new_column_name="name")
