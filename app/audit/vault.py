"""
app/audit/vault.py

Immutable Audit Vault with cryptographic chain verification.

Prevents logging tampering by chaining each event hash to the previous log
event hash. This ensures database-level edits are immediately visible as
broken hash chains.

Design:
  Hash_n = SHA256(event_type + user_id + timestamp + actor + payload_hash + Hash_{n-1})
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from typing import Any
import uuid

from sqlalchemy import select
from app.audit.event_types import AuditEventType
from app.models.layer2.audit_log import AuditLog
from app.database.session import AsyncSession

logger = logging.getLogger(__name__)


class AuditVaultManager:
    """Manages immutable audit log entries and verifies integrity chains."""

    async def log_event(
        self,
        db: AsyncSession,
        event_type: AuditEventType,
        user_id: uuid.UUID | None,
        actor: str,
        ip_address: str,
        details: str,
        payload: dict[str, Any] | None = None,
    ) -> AuditLog:
        """
        Record a cryptographically chained audit log entry.

        This method is async and must be executed in an active DB transaction.
        """
        now = datetime.now(timezone.utc)
        payload_data = payload or {}
        payload_str = json.dumps(payload_data, sort_keys=True)
        payload_hash = hashlib.sha256(payload_str.encode()).hexdigest()

        # 1. Retrieve the hash of the most recent audit log entry (genesis fallback if empty)
        stmt = select(AuditLog).order_by(AuditLog.created_at.desc()).limit(1)
        res = await db.execute(stmt)
        last_log = res.scalars().first()

        prev_hash = last_log.hash_signature if last_log else "GENESIS_HASH_0000000000000000000"

        # 2. Compute current record hash: SHA256(metadata + details + payload + prev_hash)
        components = (
            str(event_type) +
            (str(user_id) if user_id else "SYSTEM") +
            now.isoformat() +
            actor +
            payload_hash +
            prev_hash
        )
        current_hash = hashlib.sha256(components.encode()).hexdigest()

        # 3. Write record
        log_entry = AuditLog(
            event_type=str(event_type),
            user_id=user_id,
            actor=actor,
            ip_address=ip_address,
            details=details,
            payload_hash=payload_hash,
            hash_signature=current_hash,
            previous_hash=prev_hash,
            created_at=now,
        )

        db.add(log_entry)
        logger.info(
            "Audit event logged: type=%s, hash=%s, prev=%s",
            event_type, current_hash[:8], prev_hash[:8]
        )
        return log_entry

    async def verify_chain_integrity(self, db: AsyncSession) -> tuple[bool, str]:
        """
        Verify the entire audit log database verify hash chains.

        Returns:
            Tuple of (is_intact: bool, reason_or_detail: str)
        """
        stmt = select(AuditLog).order_by(AuditLog.created_at.asc())
        res = await db.execute(stmt)
        logs = res.scalars().all()

        if not logs:
            return True, "Audit log is empty (valid genesis state)."

        prev_expected_hash = "GENESIS_HASH_0000000000000000000"
        for i, log in enumerate(logs):
            # Check link to previous
            if log.previous_hash != prev_expected_hash:
                return False, f"Broken link at row {i} (ID: {log.id}): expected prev_hash '{prev_expected_hash}', got '{log.previous_hash}'."

            # Re-compute current hash to detect payload changes
            components = (
                str(log.event_type) +
                (str(log.user_id) if log.user_id else "SYSTEM") +
                log.created_at.isoformat() +
                log.actor +
                log.payload_hash +
                log.previous_hash
            )
            recomputed = hashlib.sha256(components.encode()).hexdigest()
            if log.hash_signature != recomputed:
                return False, f"Corrupted record at row {i} (ID: {log.id}): hash signature changed (calculated '{recomputed}', stored '{log.hash_signature}')."

            prev_expected_hash = log.hash_signature

        return True, f"Successfully verified {len(logs)} audit signatures without error."


# Module-level singleton
_audit_vault: AuditVaultManager | None = None


def get_audit_vault() -> AuditVaultManager:
    """Return singleton AuditVaultManager."""
    global _audit_vault
    if _audit_vault is None:
        _audit_vault = AuditVaultManager()
    return _audit_vault
