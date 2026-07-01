"""
app/cache/redis_client.py

Async Redis Connection manager supporting caching patterns.

Caches:
  - In-flight auth sessions & challenge tokens
  - Decrypted candidate lists (reduces DB access during authentication peaks)
  - Rate limiting counters

Provides an dynamic fallback in-memory dict when Redis is disabled/unavailable.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import redis.asyncio as redis

from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class RedisClient:
    """Client wrapper managing redis transactions, falling back to in-memory store."""

    def __init__(self) -> None:
        self.enabled = settings.REDIS_ENABLED
        self.url = settings.REDIS_URL
        self._client: redis.Redis | None = None
        self._fallback_store: dict[str, tuple[str, float]] = {}  # key -> (value, expiry_timestamp)

    async def connect(self) -> None:
        """Establish Redis connection pool."""
        if not self.enabled:
            logger.info("Redis is disabled, using in-memory fallback cache")
            return
        try:
            self._client = redis.from_url(
                self.url,
                encoding="utf-8",
                decode_responses=True,
                socket_timeout=2.0
            )
            # Ping test
            await self._client.ping()
            logger.info("Successfully connected to Redis at %s", self.url)
        except Exception as e:
            logger.warning("Redis connection failed (%s). Falling back to in-memory cache.", e)
            self._client = None
            self.enabled = False

    async def get(self, key: str) -> str | None:
        """Get cache entry by key."""
        if self._client and self.enabled:
            return await self._client.get(key)

        # Fallback dictionary lookup
        import time
        if key in self._fallback_store:
            val, exp = self._fallback_store[key]
            if exp == 0 or exp > time.time():
                return val
            # Expired
            del self._fallback_store[key]
        return None

    async def set(self, key: str, value: str, expire_seconds: int | None = None) -> bool:
        """Set cache entry with optional expire TTL."""
        if self._client and self.enabled:
            try:
                await self._client.set(key, value, ex=expire_seconds)
                return True
            except Exception as e:
                logger.error("Redis set operation failed: %s", e)
                return False

        # Fallback dict storage
        import time
        exp_time = (time.time() + expire_seconds) if expire_seconds else 0.0
        self._fallback_store[key] = (value, exp_time)
        return True

    async def delete(self, key: str) -> bool:
        """Evict cache entry."""
        if self._client and self.enabled:
            try:
                await self._client.delete(key)
                return True
            except Exception as e:
                logger.error("Redis delete failed: %s", e)
                return False

        self._fallback_store.pop(key, None)
        return True

    async def get_json(self, key: str) -> Any | None:
        """Get deserialized JSON cache entry."""
        val = await self.get(key)
        if val is None:
            return None
        try:
            return json.loads(val)
        except Exception:
            return None

    async def set_json(self, key: str, value: Any, expire_seconds: int | None = None) -> bool:
        """Set JSON serialized cache entry."""
        try:
            val_str = json.dumps(value)
            return await self.set(key, val_str, expire_seconds)
        except Exception as e:
            logger.error("Serialization failed for key %s: %s", key, e)
            return False


# Module-level singleton
_redis: RedisClient | None = None


async def get_redis_client() -> RedisClient:
    """Return globally cached RedisClient connection."""
    global _redis
    if _redis is None:
        _redis = RedisClient()
        await _redis.connect()
    return _redis
