"""production_layer2_upgrade

Revision ID: 20260722_0000
Revises: 2b80b82c9194
Create Date: 2026-07-22 00:00:00.000000

"""

from alembic import op
import sqlalchemy as sqla
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '20260722_0000'
down_revision = '2b80b82c9194'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # 1. Add biohash and needs_re_enrollment to voiceprints
    op.add_column('voiceprints', sqla.Column('biohash', sqla.String(length=256), nullable=True))
    op.add_column('voiceprints', sqla.Column('needs_re_enrollment', sqla.Boolean(), server_default='false', nullable=False))

    # 2. Create impostor_cohort table
    op.create_table(
        'impostor_cohort',
        sqla.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sqla.Column('embedding', postgresql.JSONB(), nullable=False),
        sqla.Column('speaker_label', sqla.String(length=100), nullable=True),
        sqla.Column('created_at', sqla.DateTime(timezone=True), nullable=False),
    )

    # 3. Create verification_audit_log table
    op.create_table(
        'verification_audit_log',
        sqla.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sqla.Column('user_id', postgresql.UUID(as_uuid=True), nullable=True, index=True),
        sqla.Column('decision', sqla.String(length=50), nullable=False),
        sqla.Column('raw_similarity', sqla.Float(), nullable=False),
        sqla.Column('snorm_score', sqla.Float(), nullable=True),
        sqla.Column('model_version', sqla.String(length=50), nullable=False),
        sqla.Column('request_id_hash', sqla.String(length=64), nullable=False),
        sqla.Column('created_at', sqla.DateTime(timezone=True), nullable=False, index=True),
    )


def downgrade() -> None:
    op.drop_table('verification_audit_log')
    op.drop_table('impostor_cohort')
    op.drop_column('voiceprints', 'needs_re_enrollment')
    op.drop_column('voiceprints', 'biohash')
