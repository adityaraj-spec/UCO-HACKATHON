"""
app/antispoof/challenge_service.py

Challenge phrase generator & validator.

Generates random challenge phrases to defeat replay attacks.
Calculates time-limited JWT challenge tokens to bind voice authentication
attempts to a specific random phrase.
"""

from __future__ import annotations

import logging
import secrets
import time
import uuid

import jwt

from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class ChallengePhraseService:
    """Generates random, cryptographically bound liveness challenges."""

    # Default fallback challenges if database is unseeded
    DEFAULT_PHRASES = [
        {"id": "PHRASE_001", "text": "Authorization code 4 9 2 0 active", "lang": "en-IN"},
        {"id": "PHRASE_002", "text": "Activate emergency backup transfer now", "lang": "en-IN"},
        {"id": "PHRASE_003", "text": "My voice security credentials are verified", "lang": "en-IN"},
        {"id": "PHRASE_004", "text": "Confirm payment of 500 rupees to wallet", "lang": "en-IN"},
        {"id": "PHRASE_005", "text": "Phase Guard system online at 10 AM", "lang": "en-IN"},
    ]

    def __init__(self) -> None:
        self.secret = settings.JWT_SECRET_KEY

    def generate_challenge(self, user_id: uuid.UUID, session_id: str) -> tuple[str, str, str]:
        """
        Generate a random challenge phrase and return bounded token.

        Args:
            user_id: Target user
            session_id: Target auth session

        Returns:
            Tuple of:
              - phrase_id
              - phrase_text
              - challenge_token (JWT bound to user/session/phrase)
        """
        phrase = secrets.choice(self.DEFAULT_PHRASES)
        phrase_id = phrase["id"]
        phrase_text = phrase["text"]

        jti = secrets.token_hex(16)
        now = int(time.time())
        exp = now + settings.CHALLENGE_TTL_SECONDS

        payload = {
            "iss": "phaseguard-l2-antispoof",
            "sub": str(user_id),
            "session_id": session_id,
            "phrase_id": phrase_id,
            "phrase_text": phrase_text,
            "iat": now,
            "exp": exp,
            "jti": jti,
        }

        token = jwt.encode(payload, self.secret, algorithm="HS256")
        logger.debug(
            "Issued challenge for user %s: session=%s, phrase_id=%s, jti=%s",
            user_id, session_id, phrase_id, jti
        )
        return phrase_id, phrase_text, token

    def verify_challenge_token(
        self,
        token: str,
        user_id: uuid.UUID,
        session_id: str,
    ) -> tuple[bool, str, str]:
        """
        Verify that a challenge token is valid, unexpired, and bound correctly.

        Returns:
            Tuple of (is_valid: bool, phrase_id: str, reason: str)
        """
        try:
            payload = jwt.decode(token, self.secret, algorithms=["HS256"])
        except jwt.ExpiredSignatureError:
            return False, "", "Challenge token has expired"
        except jwt.PyJWTError as e:
            return False, "", f"Invalid challenge token: {e}"

        # Bound constraints checks
        if payload.get("sub") != str(user_id):
            return False, "", "Token subject mismatch"
        if payload.get("session_id") != session_id:
            return False, "", "Token session binding mismatch"

        phrase_id = payload.get("phrase_id", "")
        return True, phrase_id, "OK"


# Module-level singleton
_challenge_service: ChallengePhraseService | None = None


def get_challenge_service() -> ChallengePhraseService:
    """Return singleton ChallengePhraseService."""
    global _challenge_service
    if _challenge_service is None:
        _challenge_service = ChallengePhraseService()
    return _challenge_service
