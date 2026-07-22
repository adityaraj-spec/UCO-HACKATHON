"""
app/api/v1/endpoints/challenge.py

Challenge-Response endpoint generating randomized spoken phrases for liveness checking.
Cached per session in Redis with a 60-second TTL.
"""

import random
import uuid
from fastapi import APIRouter
from pydantic import BaseModel

from app.core.config import get_settings
from app.services.redis_service import redis_service

settings = get_settings()
router = APIRouter()

CHALLENGE_PHRASES = [
    "My voice is my password in UCO Bank security",
    "Authorize payment transaction five thousand rupees",
    "Verify caller identity for mobile banking access",
    "Secure digital voice authentication system online",
    "Confirm two-factor voice verification approval code",
]


class ChallengeResponse(BaseModel):
    challenge_id: str
    phrase: str
    ttl_seconds: int


@router.get(
    "/challenge",
    response_model=ChallengeResponse,
    summary="Get a random challenge-response text phrase for liveness verification",
)
async def get_challenge() -> ChallengeResponse:
    challenge_id = str(uuid.uuid4())
    phrase = random.choice(CHALLENGE_PHRASES)

    redis_service.set(f"challenge:{challenge_id}", phrase, ex=settings.CHALLENGE_TTL_SEC)

    return ChallengeResponse(
        challenge_id=challenge_id,
        phrase=phrase,
        ttl_seconds=settings.CHALLENGE_TTL_SEC,
    )
