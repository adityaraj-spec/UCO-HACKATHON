"""
app/models/__init__.py

Imports all ORM models so that:
  1. `app.database.base.Base.metadata` is fully populated for Alembic
     autogeneration.
  2. Other modules can import models via `from app.models import User`.
"""

from app.models.user import User
from app.models.voiceprint import Voiceprint
from app.models.verification_log import VerificationLog
from app.models.risk_log import RiskLog
from app.models.kyc_enrollment import KYCEnrollmentSession, VoiceChallenge

__all__ = [
    "User",
    "Voiceprint",
    "VerificationLog",
    "RiskLog",
    "KYCEnrollmentSession",
    "VoiceChallenge",
]
