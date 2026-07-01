"""
app/middleware/replay_guard.py

FastAPI JTI replay guard dependency.

Checks claims nonces/JTIs against Redis cache database (TTL 5 mins). Ejects
requests if duplicate JTI matches to prevent authentication reply attacks.
"""

from __future__ import annotations

from fastapi import Depends, HTTPException, status
from app.cache.redis_client import get_redis_client
from app.middleware.jwt_auth import get_current_user_claims


async def check_replay_jti(
    claims: dict = Depends(get_current_user_claims)
) -> dict:
    """
    Validate that the incoming token's JTI has not been replayed.
    """
    jti = claims.get("jti")
    if not jti:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Replay check failed: token payload missing 'jti' nonce."
        )

    redis = await get_redis_client()
    cache_key = f"jwt_jti:{jti}"

    # Check if exists
    exists = await redis.get(cache_key)
    if exists:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token replay attempt detected. Request blocked by replay protection guard."
        )

    # Calculate expiration offset
    # Default cache duration is 5 minutes or remaining TTL of the token
    import time
    now = int(time.time())
    exp = claims.get("exp", now + 300)
    ttl = max(5, exp - now)

    await redis.set(cache_key, "1", expire_seconds=ttl)
    return claims
