"""
app/emergency/activation_service.py

Emergency Access Activation workflow.

Orchestrates the verification and authorization of emergency logins:
  1. Search for registered contact by verified phone number
  2. Issue OTP to verified contact
  3. Validate OTP
  4. Create temporary active EmergencyAccessEvent session bounds (default 72h max)
  5. Audit log access events
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
import uuid

from sqlalchemy import select
from app.audit.event_types import AuditEventType
from app.audit.vault import get_audit_vault
from app.core.config import get_settings
from app.emergency.contact_service import get_emergency_contact_service
from app.emergency.otp_interface import get_otp_provider
from app.models.layer2.emergency_contact import EmergencyContact, EmergencyAccessEvent
from app.database.session import AsyncSession

logger = logging.getLogger(__name__)
settings = get_settings()


class EmergencyActivationService:
    """Manages emergency session requests and execution timers."""

    def __init__(self) -> None:
        self.contact_service = get_emergency_contact_service()
        self.otp = get_otp_provider()
        self.audit = get_audit_vault()

    async def find_contact_by_phone(
        self,
        db: AsyncSession,
        user_id: uuid.UUID,
        raw_phone: str,
    ) -> EmergencyContact | None:
        """Query all active contacts for user and verify decrypted phone number match."""
        stmt = select(EmergencyContact).where(
            EmergencyContact.user_id == user_id,
            EmergencyContact.is_active == True,
            EmergencyContact.is_verified == True
        )
        res = await db.execute(stmt)
        contacts = res.scalars().all()

        for c in contacts:
            ph, _ = await self.contact_service.decrypt_contact_pii(c)
            if ph == raw_phone:
                return c

        return None

    async def initiate_activation(
        self,
        db: AsyncSession,
        contact: EmergencyContact,
    ) -> str:
        """Trigger liveness activation OTP code to verified contact's phone."""
        prov = self.otp
        phone, _ = await self.contact_service.decrypt_contact_pii(contact)
        otp = await prov.generate_and_send(phone)
        return otp

    async def complete_activation(
        self,
        db: AsyncSession,
        contact: EmergencyContact,
        otp_code: str,
        caller_ip: str,
    ) -> EmergencyAccessEvent | None:
        """
        Verify activation SMS OTP and spawn a 72-hour access event record.
        """
        phone, _ = await self.contact_service.decrypt_contact_pii(contact)
        verified = await self.otp.verify_otp(phone, otp_code)

        if not verified:
            logger.warning(
                "Emergency activation OTP verification failed for contact ID %s",
                contact.id
            )
            return None

        # 1. Spawn access event session with 72h max lifetime
        now = datetime.now(timezone.utc)
        expires = now + timedelta(hours=settings.EMERGENCY_ACCESS_TTL_HOURS)

        import secrets
        access_event = EmergencyAccessEvent(
            emergency_contact_id=contact.id,
            user_id=contact.user_id,
            activation_reason="Emergency override activation requested by contact",
            verification_method_used="SMS_OTP",
            access_token_jti=secrets.token_hex(16),
            access_scope_granted=contact.access_scope,
            ip_address=caller_ip,
            status="ACTIVE",
            activated_at=now,
            expires_at=expires,
        )

        db.add(access_event)
        await db.flush()

        # 2. Add entry to Immutable Audit Vault
        await self.audit.log_event(
            db=db,
            event_type=AuditEventType.EMERGENCY_ACTIVATE,
            user_id=contact.user_id,
            actor=f"EMERGENCY_CONTACT_{contact.contact_name}",
            ip_address=caller_ip,
            details=f"Emergency override activated by trustee contact. Scope: {contact.access_scope}.",
            payload={
                "contact_id": str(contact.id),
                "scope": contact.access_scope,
                "expires_at": expires.isoformat(),
            }
        )

        logger.info(
            "Emergency activation COMPLETE: user=%s, contact=%s, expires=%s",
            contact.user_id, contact.contact_name, expires,
        )
        return access_event

    async def verify_active_emergency_session(
        self,
        db: AsyncSession,
        user_id: uuid.UUID,
    ) -> EmergencyAccessEvent | None:
        """
        Check if there is currently an active, unexpired emergency override session.
        """
        now = datetime.now(timezone.utc)
        stmt = select(EmergencyAccessEvent).where(
            EmergencyAccessEvent.user_id == user_id,
            EmergencyAccessEvent.status == "ACTIVE",
            EmergencyAccessEvent.expires_at > now
        )
        res = await db.execute(stmt)
        event = res.scalars().first()

        if event:
            return event

        # Auto-expire any stale active sessions
        stmt_stale = select(EmergencyAccessEvent).where(
            EmergencyAccessEvent.user_id == user_id,
            EmergencyAccessEvent.status == "ACTIVE",
            EmergencyAccessEvent.expires_at <= now
        )
        res_stale = await db.execute(stmt_stale)
        stale_events = res_stale.scalars().all()
        for se in stale_events:
            se.status = "EXPIRED"
            # Log expiration in audit
            await self.audit.log_event(
                db=db,
                event_type=AuditEventType.EMERGENCY_EXPIRE,
                user_id=se.user_id,
                actor="SYSTEM_SCHEDULER",
                ip_address="127.0.0.1",
                details=f"Emergency access event {se.id} expired automatically after 72 hours.",
                payload={"event_id": str(se.id)}
            )

        return None


# Module-level singleton
_activation_service: EmergencyActivationService | None = None


def get_emergency_activation_service() -> EmergencyActivationService:
    """Return singleton EmergencyActivationService."""
    global _activation_service
    if _activation_service is None:
        _activation_service = EmergencyActivationService()
    return _activation_service
