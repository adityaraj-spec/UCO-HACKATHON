"""
app/emergency/otp_interface.py

OTP Integration engine.

Handles standard SMS verification for emergency contacts during registration
and activation verification. Includes dummy mock provider for testing.
"""

from abc import ABC, abstractmethod
import logging
import secrets
from app.cache.redis_client import get_redis_client

logger = logging.getLogger(__name__)


class OTPProvider(ABC):
    """Abstract interface for OTP delivery (SMS/Email)."""

    @abstractmethod
    async def send_otp(self, phone: str, otp: str, message: str) -> bool:
        """Send formatted OTP code to destination number."""

    @abstractmethod
    async def verify_otp(self, phone: str, user_provided_otp: str) -> bool:
        """Verify OTP against the cache."""


class MockOTPProvider(OTPProvider):
    """Mock OTP provider storing codes in Redis cache with 5 minute expiration."""

    def __init__(self) -> None:
        self.ttl = 300  # 5 minutes

    async def generate_and_send(self, phone: str) -> str:
        """Generate a random 6-digit numeric OTP and send it (mocked)."""
        otp = "".join(secrets.choice("0123456789") for _ in range(6))
        message = f"PhaseGuard Verification Code: {otp}. Valid for 5 mins."
        
        await self.send_otp(phone, otp, message)
        return otp

    async def send_otp(self, phone: str, otp: str, message: str) -> bool:
        redis_client = await get_redis_client()
        cache_key = f"otp:{phone}"
        await redis_client.set(cache_key, otp, expire_seconds=self.ttl)
        
        logger.info("[MOCK SMS] Sent to: %s | Message: %s", phone, message)
        return True

    async def verify_otp(self, phone: str, user_provided_otp: str) -> bool:
        redis_client = await get_redis_client()
        cache_key = f"otp:{phone}"
        stored = await redis_client.get(cache_key)
        
        if stored and stored == user_provided_otp:
            # Revoke to prevent double validation
            await redis_client.delete(cache_key)
            return True
        return False


# Module-level singleton
_otp_provider: OTPProvider | None = None


def get_otp_provider() -> OTPProvider:
    """Return configured OTP provider."""
    global _otp_provider
    if _otp_provider is None:
        _otp_provider = MockOTPProvider()
    return _otp_provider
