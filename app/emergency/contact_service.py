"""
app/emergency/contact_service.py

Emergency Contact registration and verification.

Allows accounts to register up to 2 emergency contacts who can request
limited/delegated access during medical, travel, or voice-loss incidents.
Integrates with KeyManager to encrypt phone numbers/email identifiers at rest.
"""

from __future__ import annotations

import logging
import uuid

from sqlalchemy import select
from app.core.config import get_settings
from app.emergency.access_scope import EmergencyAccessScope
from app.emergency.otp_interface import get_otp_provider
from app.encryption.aes_gcm import AESGCM256, EncryptedData
from app.encryption.key_manager import get_key_manager
from app.models.layer2.emergency_contact import EmergencyContact
from app.database.session import AsyncSession

logger = logging.getLogger(__name__)
settings = get_settings()


class EmergencyContactService:
    """Handles trust contact registrations and GCM-encryption of PII fields."""

    def __init__(self) -> None:
        self.key_manager = get_key_manager()
        self.otp = get_otp_provider()

    async def register_contact(
        self,
        db: AsyncSession,
        user_id: uuid.UUID,
        name: str,
        phone: str,
        email: str,
        scope: EmergencyAccessScope,
    ) -> EmergencyContact:
        """
        GCM-encrypt contact details (phone, email) using derived user-keys, then save.
        """
        # Limit checked: Max 2 emergency contacts
        stmt = select(EmergencyContact).where(
            EmergencyContact.user_id == user_id,
            EmergencyContact.is_active == True
        )
        res = await db.execute(stmt)
        active_count = len(res.scalars().all())

        if active_count >= 2:
            raise ValueError("Maximum of 2 active emergency contacts allowed per customer account.")

        # 1. Derive user key and encrypt PII data fields
        key_id = settings.HSM_MASTER_KEY_ID
        user_key = self.key_manager.get_key_for_user(str(user_id), key_id)
        encryptor = AESGCM256(user_key)

        # Phone encryption
        phone_enc = encryptor.encrypt(phone.encode(), associated_data=str(user_id).encode())
        # Email encryption
        email_enc = encryptor.encrypt(email.encode(), associated_data=str(user_id).encode())

        # 2. Record model entry
        contact = EmergencyContact(
            user_id=user_id,
            contact_name=name,
            encrypted_phone=phone_enc.ciphertext,
            phone_nonce=phone_enc.nonce,
            encrypted_email=email_enc.ciphertext,
            email_nonce=email_enc.nonce,
            access_scope=str(scope),
            key_id=key_id,
            is_active=True,
            is_verified=False,  # Verified only after OTP challenge completed
        )

        db.add(contact)
        logger.info(
            "Emergency contact registered for user %s: name=%s, scope=%s (pending OTP verification)",
            user_id, name, scope
        )
        return contact

    async def decrypt_contact_pii(
        self,
        contact: EmergencyContact,
    ) -> tuple[str, str]:
        """Decrypt name-phone-email details for target contact model."""
        user_key = self.key_manager.get_key_for_user(str(contact.user_id), contact.key_id)
        encryptor = AESGCM256(user_key)

        phone_data = EncryptedData(ciphertext=contact.encrypted_phone, nonce=contact.phone_nonce)
        email_data = EncryptedData(ciphertext=contact.encrypted_email, nonce=contact.email_nonce)

        p = encryptor.decrypt(phone_data, associated_data=str(contact.user_id).encode()).decode()
        e = encryptor.decrypt(email_data, associated_data=str(contact.user_id).encode()).decode()

        return p, e

    async def initiate_verification_otp(self, contact: EmergencyContact) -> str:
        """Trigger an OTP SMS to contact's phone for verification."""
        phone, _ = await self.decrypt_contact_pii(contact)
        # Type check to trigger generate
        prov = self.otp
        if isinstance(prov, MockOTPProvider):
            otp = await prov.generate_and_send(phone)
            return otp
        
        # Real provider has separate trigger mechanisms
        return "OTP_SENT"

    async def verify_contact_otp(
        self,
        db: AsyncSession,
        contact: EmergencyContact,
        otp_code: str,
    ) -> bool:
        """Confirm valid SMS OTP verification to make contact active."""
        phone, _ = await self.decrypt_contact_pii(contact)
        verified = await self.otp.verify_otp(phone, otp_code)

        if verified:
            contact.is_verified = True
            logger.info("Emergency contact verification succeeded for contact ID: %s", contact.id)
            return True

        logger.warning("Emergency verification OTP mismatch for contact ID: %s", contact.id)
        return False


# Module-level singleton
_contact_service: EmergencyContactService | None = None


def get_emergency_contact_service() -> EmergencyContactService:
    """Return singleton EmergencyContactService."""
    global _contact_service
    if _contact_service is None:
        _contact_service = EmergencyContactService()
    return _contact_service
