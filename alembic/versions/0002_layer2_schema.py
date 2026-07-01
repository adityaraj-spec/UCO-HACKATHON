"""create layer 2 schema (enrollments, anchor_embeddings, rolling_embeddings, auth_sessions, auth_history, consents, audit_logs, challenge_phrases, emergency_contacts, emergency_access_log, user_thresholds, voice_metadata)

Revision ID: 0002_layer2_schema
Revises: 0001_initial_schema
Create Date: 2026-06-30 00:00:00

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "0002_layer2_schema"
down_revision: Union[str, None] = "0001_initial_schema"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. enrollments
    op.create_table(
        "enrollments",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False, server_default="IN_PROGRESS"),
        sa.Column("samples_submitted", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("samples_required", sa.Integer(), nullable=False, server_default="5"),
        sa.Column("created_by_ip", sa.String(length=45), nullable=True),
        sa.Column("created_by_user_agent", sa.String(length=255), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_enrollments_user_id", "enrollments", ["user_id"])

    # 2. anchor_embeddings
    op.create_table(
        "anchor_embeddings",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False, unique=True),
        sa.Column("encrypted_embedding", sa.LargeBinary(), nullable=False),
        sa.Column("encryption_nonce", sa.LargeBinary(12), nullable=False),
        sa.Column("embedding_hash", sa.String(length=64), nullable=False),
        sa.Column("key_id", sa.String(length=128), nullable=False),
        sa.Column("template_version", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default="true"),
        sa.Column("enrollment_quality_score", sa.Float(), nullable=True),
        sa.Column("recording_count", sa.Integer(), nullable=False, server_default="5"),
        sa.Column("embedding_dim", sa.Integer(), nullable=False, server_default="192"),
        sa.Column("enrolled_via", sa.String(length=32), nullable=False, server_default="MOBILE_APP"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("deactivated_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_anchor_embeddings_user_id", "anchor_embeddings", ["user_id"])

    # 3. rolling_embeddings
    op.create_table(
        "rolling_embeddings",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("encrypted_embedding", sa.LargeBinary(), nullable=False),
        sa.Column("encryption_nonce", sa.LargeBinary(12), nullable=False),
        sa.Column("embedding_hash", sa.String(length=64), nullable=False),
        sa.Column("key_id", sa.String(length=128), nullable=False),
        sa.Column("template_version", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("pool_position", sa.Integer(), nullable=False),
        sa.Column("auth_score", sa.Float(), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default="true"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_rolling_embeddings_user_id", "rolling_embeddings", ["user_id"])

    # 4. auth_sessions
    op.create_table(
        "auth_sessions",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("phrase_id", sa.String(length=64), nullable=False),
        sa.Column("phrase_text", sa.String(length=255), nullable=False),
        sa.Column("challenge_token", sa.String(length=512), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False, server_default="PENDING"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_auth_sessions_user_id", "auth_sessions", ["user_id"])

    # 5. auth_history
    op.create_table(
        "auth_history",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("session_id", sa.String(length=128), nullable=False),
        sa.Column("phrase_id", sa.String(length=64), nullable=False),
        sa.Column("attempt_number", sa.Integer(), nullable=False),
        sa.Column("similarity_score", sa.Float(), nullable=False),
        sa.Column("is_similarity_passed", sa.Boolean(), nullable=False),
        sa.Column("liveness_score", sa.Float(), nullable=False),
        sa.Column("is_liveness_passed", sa.Boolean(), nullable=False),
        sa.Column("risk_score", sa.Float(), nullable=False),
        sa.Column("risk_level", sa.String(length=32), nullable=False),
        sa.Column("final_decision", sa.String(length=32), nullable=False),
        sa.Column("failure_reason", sa.String(length=255), nullable=True),
        sa.Column("ip_device_trusted", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_auth_history_user_id", "auth_history", ["user_id"])

    # 6. consents
    op.create_table(
        "consents",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default="true"),
        sa.Column("ip_address", sa.String(length=45), nullable=False),
        sa.Column("user_agent", sa.String(length=255), nullable=False),
        sa.Column("consent_type", sa.String(length=64), nullable=False, server_default="EXPLICIT_OPT_IN"),
        sa.Column("consent_token", sa.String(length=255), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("withdrawn_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_consents_user_id", "consents", ["user_id"])

    # 7. audit_logs
    op.create_table(
        "audit_logs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("event_type", sa.String(length=64), nullable=False),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("actor", sa.String(length=64), nullable=False),
        sa.Column("ip_address", sa.String(length=45), nullable=False),
        sa.Column("details", sa.String(length=512), nullable=False),
        sa.Column("hash_signature", sa.String(length=64), nullable=False),
        sa.Column("previous_hash", sa.String(length=64), nullable=False),
        sa.Column("payload", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_audit_logs_user_id", "audit_logs", ["user_id"])

    # 8. challenge_phrases
    op.create_table(
        "challenge_phrases",
        sa.Column("phrase_id", sa.String(length=64), primary_key=True),
        sa.Column("phrase_text", sa.String(length=255), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default="true"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )

    # 9. emergency_contacts
    op.create_table(
        "emergency_contacts",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("contact_name", sa.String(length=255), nullable=False),
        sa.Column("encrypted_phone", sa.LargeBinary(), nullable=False),
        sa.Column("encrypted_email", sa.LargeBinary(), nullable=False),
        sa.Column("is_verified", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("access_scope", sa.String(length=32), nullable=False, server_default="READ_ONLY"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_emergency_contacts_user_id", "emergency_contacts", ["user_id"])

    # 10. emergency_access_log
    op.create_table(
        "emergency_access_log",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("contact_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("access_scope", sa.String(length=32), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False, server_default="ACTIVE"),
        sa.Column("ip_address", sa.String(length=45), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_emergency_access_log_user_id", "emergency_access_log", ["user_id"])

    # 11. user_thresholds
    op.create_table(
        "user_thresholds",
        sa.Column("user_id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("default_threshold", sa.Float(), nullable=False, server_default="0.82"),
        sa.Column("current_threshold", sa.Float(), nullable=False, server_default="0.82"),
        sa.Column("consecutive_failures", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("illness_window_active", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("illness_window_start", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_successful_auth_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )

    # 12. voice_metadata
    op.create_table(
        "voice_metadata",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("recording_id", sa.String(length=64), nullable=False),
        sa.Column("duration_seconds", sa.Float(), nullable=False),
        sa.Column("snr_db", sa.Float(), nullable=False),
        sa.Column("voice_ratio", sa.Float(), nullable=False),
        sa.Column("codec", sa.String(length=32), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_voice_metadata_user_id", "voice_metadata", ["user_id"])


def downgrade() -> None:
    op.drop_index("ix_voice_metadata_user_id", table_name="voice_metadata")
    op.drop_table("voice_metadata")

    op.drop_table("user_thresholds")

    op.drop_index("ix_emergency_access_log_user_id", table_name="emergency_access_log")
    op.drop_table("emergency_access_log")

    op.drop_index("ix_emergency_contacts_user_id", table_name="emergency_contacts")
    op.drop_table("emergency_contacts")

    op.drop_table("challenge_phrases")

    op.drop_index("ix_audit_logs_user_id", table_name="audit_logs")
    op.drop_table("audit_logs")

    op.drop_index("ix_consents_user_id", table_name="consents")
    op.drop_table("consents")

    op.drop_index("ix_auth_history_user_id", table_name="auth_history")
    op.drop_table("auth_history")

    op.drop_index("ix_auth_sessions_user_id", table_name="auth_sessions")
    op.drop_table("auth_sessions")

    op.drop_index("ix_rolling_embeddings_user_id", table_name="rolling_embeddings")
    op.drop_table("rolling_embeddings")

    op.drop_index("ix_anchor_embeddings_user_id", table_name="anchor_embeddings")
    op.drop_table("anchor_embeddings")

    op.drop_index("ix_enrollments_user_id", table_name="enrollments")
    op.drop_table("enrollments")
