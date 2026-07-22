"""
Enrollment access controls for the Video KYC, secure mailer, and branch paths.

This module deliberately avoids schema changes. Known enrollment devices are
tracked through RedisService, which already provides an in-memory fallback for
local demos.
"""

import uuid
from dataclasses import dataclass

from app.services.redis_service import redis_service
from app.utils.exceptions import EnrollmentRejectedError

CHANNEL_DIRECT_API = "DIRECT_API"
CHANNEL_VIDEO_KYC = "VIDEO_KYC"
CHANNEL_SECURE_MAILER = "SECURE_MAILER"
CHANNEL_BRANCH = "BRANCH"

_KNOWN_DEVICE_TTL_SECONDS = 365 * 24 * 60 * 60


@dataclass(frozen=True)
class EnrollmentAccessContext:
    channel: str = CHANNEL_DIRECT_API
    biometric_consent_confirmed: bool = False
    identity_confirmed: bool = False
    authenticated: bool = False
    otp_verified: bool = False
    device_id: str | None = None
    branch_officer_id: str | None = None


def normalize_channel(channel: str | None) -> str:
    return (channel or CHANNEL_DIRECT_API).strip().upper()


class EnrollmentAccessService:
    """Validates enrollment channel requirements without changing /enroll semantics."""

    def validate(
        self,
        *,
        user_id: uuid.UUID,
        has_existing_voiceprint: bool,
        context: EnrollmentAccessContext,
    ) -> None:
        channel = normalize_channel(context.channel)

        if channel == CHANNEL_VIDEO_KYC and not context.biometric_consent_confirmed:
            raise EnrollmentRejectedError(
                "Explicit biometric consent is required for VIDEO_KYC enrollment."
            )

        if channel == CHANNEL_SECURE_MAILER:
            if not context.authenticated or not context.otp_verified:
                raise EnrollmentRejectedError(
                    "Secure mailer enrollment requires successful login and OTP before recording."
                )
            if not context.biometric_consent_confirmed:
                raise EnrollmentRejectedError(
                    "Explicit biometric consent is required before secure mailer enrollment."
                )

        if channel == CHANNEL_BRANCH:
            if not context.identity_confirmed or not context.branch_officer_id:
                raise EnrollmentRejectedError(
                    "Branch enrollment requires officer identity confirmation before recording."
                )

        if has_existing_voiceprint and context.device_id:
            known_key = self._device_key(user_id, context.device_id)
            if redis_service.get(known_key) != "known" and not context.otp_verified:
                raise EnrollmentRejectedError(
                    "Re-enrollment from an unrecognized device requires fresh OTP confirmation."
                )

    def remember_device(self, user_id: uuid.UUID, device_id: str | None) -> None:
        if not device_id:
            return
        redis_service.set(self._device_key(user_id, device_id), "known", ex=_KNOWN_DEVICE_TTL_SECONDS)

    @staticmethod
    def _device_key(user_id: uuid.UUID, device_id: str) -> str:
        return f"enrollment_device:{user_id}:{device_id}"
