"""
app/services/redis_service.py

Redis client service for session caching, JWT replay protection, and rate limiting.
Implements a graceful in-memory fallback if Redis is unavailable or fails.
"""

import time
import logging
import threading
from typing import Optional, Union, Dict, List
import redis

from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

class RedisService:
    def __init__(self):
        self.redis_client: Optional[redis.Redis] = None
        self.is_connected = False
        self.last_connect_attempt = 0.0
        self.connect_cooldown = 30.0  # seconds to wait before retrying connection
        
        # In-memory fallback structures
        self._in_memory_cache: Dict[str, tuple[str, float]] = {}  # key -> (value, expiry_time)
        self._in_memory_rates: Dict[str, List[float]] = {}       # key -> list of timestamps
        self._fallback_lock = threading.Lock()
        
        self._connect()

    def _connect(self):
        """Attempts to connect to Redis, logs fallback mode if unavailable."""
        self.last_connect_attempt = time.time()
        try:
            self.redis_client = redis.Redis(
                host=settings.REDIS_HOST,
                port=settings.REDIS_PORT,
                db=settings.REDIS_DB,
                decode_responses=True,
                socket_connect_timeout=2.0,
                socket_timeout=2.0
            )
            # Ping to confirm connection
            self.redis_client.ping()
            self.is_connected = True
            logger.info("Successfully connected to Redis.")
        except Exception as e:
            self.is_connected = False
            self.redis_client = None
            logger.warning(
                f"Redis is unavailable: {e}. "
                "PhaseGuard is running with in-memory fallback for rate limiting and cache."
            )

    def _ensure_connection(self) -> bool:
        """Helper to check connection and attempt reconnect if was offline, respecting cooldown."""
        if not self.is_connected:
            # Only retry if cooldown has expired
            if time.time() - self.last_connect_attempt > self.connect_cooldown:
                logger.info("Redis cooldown expired. Attempting to reconnect...")
                self._connect()
        return self.is_connected

    # --- Cache get/set ---
    def get(self, key: str) -> Optional[str]:
        """Gets a string value by key."""
        if self._ensure_connection():
            try:
                return self.redis_client.get(key)
            except Exception as e:
                logger.error(f"Redis get failed: {e}. Falling back to in-memory.")
                
        # In-memory fallback
        with self._fallback_lock:
            val_tuple = self._in_memory_cache.get(key)
            if val_tuple:
                value, expiry = val_tuple
                if expiry is None or expiry > time.time():
                    return value
                else:
                    del self._in_memory_cache[key]  # Clean up expired
            return None

    def set(self, key: str, value: str, ex: Optional[int] = None) -> bool:
        """Sets a string value with optional expiration in seconds."""
        if self._ensure_connection():
            try:
                return self.redis_client.set(key, value, ex=ex)
            except Exception as e:
                logger.error(f"Redis set failed: {e}. Falling back to in-memory.")
                
        # In-memory fallback
        with self._fallback_lock:
            expiry = time.time() + ex if ex is not None else None
            self._in_memory_cache[key] = (value, expiry)
            return True

    # --- JWT Replay Protection ---
    def mark_token_used(self, jti: str, ttl_seconds: int) -> bool:
        """
        Registers a JTI (JWT token ID) as used.
        Returns True if the token was NOT used before (and was marked successfully).
        Returns False if the token was already used (replay attack detected!).
        """
        key = f"jwt_jti:{jti}"
        if self._ensure_connection():
            try:
                # Set NX returns True if key was set (did not exist)
                return bool(self.redis_client.set(key, "used", ex=ttl_seconds, nx=True))
            except Exception as e:
                logger.error(f"Redis mark_token_used failed: {e}. Falling back to in-memory.")
                
        # In-memory fallback
        with self._fallback_lock:
            now = time.time()
            val_tuple = self._in_memory_cache.get(key)
            if val_tuple:
                _, expiry = val_tuple
                if expiry > now:
                    logger.warning(f"Replay attack detected in-memory for JTI: {jti}")
                    return False
            
            # Not used or expired, mark it
            expiry = now + ttl_seconds
            self._in_memory_cache[key] = ("used", expiry)
            return True

    # --- Rate Limiting (Sliding Window Log) ---
    def is_rate_limited(self, key: str, limit: int, window_seconds: int) -> bool:
        """
        Checks if a key (e.g. user_id, ip) exceeds the rate limit.
        Implements a sliding window log.
        Returns True if rate limited, False if request is allowed.
        """
        now = time.time()
        redis_key = f"rate_limit:{key}"
        
        if self._ensure_connection():
            try:
                pipe = self.redis_client.pipeline()
                # Remove timestamps older than window
                pipe.zremrangebyscore(redis_key, 0, now - window_seconds)
                # Add current request timestamp
                pipe.zadd(redis_key, {str(now): now})
                # Get total requests in this window
                pipe.zcard(redis_key)
                # Set TTL on key to prevent leakage
                pipe.expire(redis_key, window_seconds)
                
                _, _, count, _ = pipe.execute()
                
                if count > limit:
                    logger.warning(f"Rate limit exceeded for {key} in Redis: {count}/{limit}")
                    return True
                return False
            except Exception as e:
                logger.error(f"Redis rate limiting failed: {e}. Falling back to in-memory.")

        # In-memory fallback
        with self._fallback_lock:
            timestamps = self._in_memory_rates.get(redis_key, [])
            # Filter timestamps in current window
            valid_cutoff = now - window_seconds
            timestamps = [ts for ts in timestamps if ts > valid_cutoff]
            
            timestamps.append(now)
            self._in_memory_rates[redis_key] = timestamps
            
            if len(timestamps) > limit:
                logger.warning(f"Rate limit exceeded for {key} in-memory: {len(timestamps)}/{limit}")
                return True
            return False

    # --- Audio Perceptual Replay Protection ---
    def check_and_store_audio_hash(self, user_id: str, audio_hash: str, ttl_sec: int = 86400) -> bool:
        """
        Check if an audio perceptual hash has been submitted by this user in the last 24 hours.
        Returns True if fresh (not replayed), False if REPLAY_DETECTED.
        """
        key = f"audio_hash:{user_id}:{audio_hash}"
        return self.mark_token_used(key, ttl_seconds=ttl_sec)


# Singleton instance
redis_service = RedisService()
