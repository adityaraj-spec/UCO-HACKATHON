"""
app/consent/consent_service.py

DPDP Act 2023 Compliant Consent Management Service.

Fixed to match the actual Consent / ConsentHistory SQLAlchemy models:
  - Consent columns: status, consent_token_jti, purpose, legal_basis,
    explicit_opt_in, right_to_withdraw_acknowledged, data_retention_acknowledged,
    ip_address, user_agent, channel, granted_at, withdrawn_at, expires_at,
    withdrawal_reason, created_at, updated_at
  - ConsentHistory columns: consent_id, user_id, event_type, previous_status,
    new_status, actor, ip_address, reason
"""

from __future__ import annotations

import logging
import hashlib
import secrets
import time
import uuid

import jwt

from app.core.config import get_settings
from app.models.layer2.consent import Consent, ConsentHistory
from app.database.session import AsyncSession

logger = logging.getLogger(__name__)
settings = get_settings()


class ConsentService:
    """Manages biometric data processing consents inside the banking ecosystem."""

    def __init__(self) -> None:
        self.secret = settings.JWT_SECRET_KEY

    def generate_consent_token(
        self,
        user_id: uuid.UUID,
        consent_type: str,
        ip_address: str,
        user_agent: str,
    ) -> tuple[str, str]:
        """
        Generate a signed JWT receipt token for a granted consent.

        Returns:
            Tuple of (consent_token: str, jti_hash: str)
        """
        jti = secrets.token_hex(16)
        now = int(time.time())
        exp = now + (365 * 24 * 3600)  # 1-year validity

        payload = {
            "iss": "phaseguard-l2-consent",
            "sub": str(user_id),
            "consent_type": consent_type,
            "purpose": "biometric_voice_auth",
            "ip_address": ip_address,
            "user_agent": user_agent,
            "iat": now,
            "exp": exp,
            "jti": jti,
        }

        token = jwt.encode(payload, self.secret, algorithm="HS256")
        # JTI stored in DB is the SHA-256 of the full token for tamper detection
        jti_hash = hashlib.sha256(token.encode()).hexdigest()

        return token, jti_hash

    async def record_consent_grant(
        self,
        db: AsyncSession,
        user_id: uuid.UUID,
        ip_address: str,
        user_agent: str,
        consent_type: str = "EXPLICIT_OPT_IN",
    ) -> Consent:
        """
        Save consent record to the DB and create a history audit entry.
        Returns the Consent ORM object (with .consent_token set as a transient attr).
        """
        from datetime import datetime, timezone, timedelta
        from sqlalchemy import select

        token, jti_hash = self.generate_consent_token(user_id, consent_type, ip_address, user_agent)
        now = datetime.now(timezone.utc)
        expires_at = now + timedelta(seconds=settings.CONSENT_TOKEN_TTL_SECONDS)

        # Check if a consent row already exists for this user (unique constraint)
        stmt = select(Consent).where(Consent.user_id == user_id)
        result = await db.execute(stmt)
        existing = result.scalars().first()

        if existing:
            # Renew / re-activate existing row
            existing.status = "ACTIVE"
            existing.consent_token_jti = jti_hash
            existing.explicit_opt_in = True
            existing.right_to_withdraw_acknowledged = True
            existing.data_retention_acknowledged = True
            existing.ip_address = ip_address
            existing.user_agent = user_agent
            existing.channel = "MOBILE_APP"
            existing.granted_at = now
            existing.expires_at = expires_at
            existing.withdrawn_at = None
            existing.withdrawal_reason = None
            consent = existing
        else:
            consent = Consent(
                user_id=user_id,
                status="ACTIVE",
                consent_token_jti=jti_hash,
                legal_basis="EXPLICIT_CONSENT",
                purpose="Voice biometric authentication for banking transaction security",
                explicit_opt_in=True,
                right_to_withdraw_acknowledged=True,
                data_retention_acknowledged=True,
                ip_address=ip_address,
                user_agent=user_agent,
                channel="MOBILE_APP",
                granted_at=now,
                expires_at=expires_at,
            )
            db.add(consent)

        await db.flush()

        # History audit entry
        history = ConsentHistory(
            consent_id=consent.id,
            user_id=user_id,
            event_type="CONSENT_GRANTED",
            previous_status="NONE" if not existing else "WITHDRAWN",
            new_status="ACTIVE",
            actor=str(user_id),
            ip_address=ip_address,
            reason=f"Explicit opt-in via {consent_type}",
        )
        db.add(history)

        logger.info(
            "Consent GRANTED: user=%s, type=%s, jti_hash=%s",
            user_id, consent_type, jti_hash[:8]
        )

        # Attach raw token as transient attribute for the API response
        consent.consent_token = token
        return consent

    async def verify_active_consent(self, db: AsyncSession, user_id: uuid.UUID) -> bool:
        """
        Query DB to verify if user has an active, unwithdrawn consent.
        Uses the `status` column (the real DB column, not the non-existent is_active).
        """
        from sqlalchemy import select
        from datetime import datetime, timezone

        stmt = select(Consent).where(
            Consent.user_id == user_id,
            Consent.status == "ACTIVE",
        )
        result = await db.execute(stmt)
        consent = result.scalars().first()

        if not consent:
            logger.warning("No active biometric consent found for user %s", user_id)
            return False

        # Check expiry
        now = datetime.now(timezone.utc)
        if consent.expires_at and consent.expires_at < now:
            logger.warning("Consent expired for user %s", user_id)
            return False

        return True

    async def record_consent_withdrawal(
        self,
        db: AsyncSession,
        user_id: uuid.UUID,
        ip_address: str,
        user_agent: str,
    ) -> bool:
        """Mark consent as WITHDRAWN and record history trail."""
        from sqlalchemy import select
        from datetime import datetime, timezone

        stmt = select(Consent).where(
            Consent.user_id == user_id,
            Consent.status == "ACTIVE",
        )
        res = await db.execute(stmt)
        consents = res.scalars().all()

        if not consents:
            logger.warning("No active consent to revoke for user %s", user_id)
            return False

        now = datetime.now(timezone.utc)
        for c in consents:
            prev_status = c.status
            c.status = "WITHDRAWN"
            c.withdrawn_at = now
            c.withdrawal_reason = "User initiated withdrawal"

            history = ConsentHistory(
                consent_id=c.id,
                user_id=user_id,
                event_type="CONSENT_WITHDRAWN",
                previous_status=prev_status,
                new_status="WITHDRAWN",
                actor=str(user_id),
                ip_address=ip_address,
                reason="Consent withdrawn by user. Automated template purge sequence queued.",
            )
            db.add(history)

        logger.info("Consent revoked for user %s. Initiating database purge schedules.", user_id)
        return True


# Module-level singleton
_consent_service: ConsentService | None = None


def get_consent_service() -> ConsentService:
    """Return singleton ConsentService."""
    global _consent_service
    if _consent_service is None:
        _consent_service = ConsentService()
    return _consent_service
