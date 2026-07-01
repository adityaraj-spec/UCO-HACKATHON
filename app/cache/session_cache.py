"""
app/cache/session_cache.py

Session cache helpers.
Manages session and challenge token lifecycle in redis.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from app.cache.redis_client import get_redis_client
from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class SessionCache:
    """Orchestrates session, challenge, and lock caching in Redis."""

    def __init__(self) -> None:
        self.session_ttl = settings.REDIS_SESSION_TTL_SECONDS
        self.challenge_ttl = settings.REDIS_CHALLENGE_TTL_SECONDS

    async def save_auth_session(self, session_id: str, session_data: dict[str, Any]) -> bool:
        """Cache current in-flight authentication session state."""
        key = f"auth_session:{session_id}"
        client = await get_redis_client()
        return await client.set_json(key, session_data, self.session_ttl)

    async def get_auth_session(self, session_id: str) -> dict[str, Any] | None:
        """Fetch active session state."""
        key = f"auth_session:{session_id}"
        client = await get_redis_client()
        return await client.get_json(key)

    async def delete_auth_session(self, session_id: str) -> bool:
        """Invalidate session."""
        key = f"auth_session:{session_id}"
        client = await get_redis_client()
        return await client.delete(key)

    async def register_challenge_token(self, jti: str, phrase_id: str, user_id: uuid.UUID) -> bool:
        """Mark challenge JTI token as active to prevent replay reuse."""
        key = f"challenge_jti:{jti}"
        val = {
            "phrase_id": phrase_id,
            "user_id": str(user_id),
            "status": "ACTIVE"
        }
        client = await get_redis_client()
        return await client.set_json(key, val, self.challenge_ttl)

    async def verify_and_consume_jti(self, jti: str) -> bool:
        """
        Verify JTI is active and immediately revoke/consume it.

        This guarantees single-use challenge phrases, preventing replay.
        """
        key = f"challenge_jti:{jti}"
        client = await get_redis_client()
        data = await client.get_json(key)
        if data and data.get("status") == "ACTIVE":
            # Consume token
            await client.delete(key)
            return True
        return False
        
    async def get_cached_embeddings(self, user_id: uuid.UUID) -> list[list[float]] | None:
        """Retrieve user's rolling templates from cache."""
        key = f"rolling_templates:{user_id}"
        client = await get_redis_client()
        return await client.get_json(key)

    async def cache_embeddings(self, user_id: uuid.UUID, templates: list[list[float]]) -> bool:
        """Cache user's active rolling templates."""
        key = f"rolling_templates:{user_id}"
        client = await get_redis_client()
        return await client.set_json(key, templates, settings.REDIS_EMBEDDING_CACHE_TTL_SECONDS)

    async def invalidate_embeddings_cache(self, user_id: uuid.UUID) -> bool:
        """Evict embeddings cache on rolling updates."""
        key = f"rolling_templates:{user_id}"
        client = await get_redis_client()
        return await client.delete(key)
