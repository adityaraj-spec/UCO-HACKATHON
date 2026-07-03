"""
app/middleware/rate_limiter.py

FastAPI rate limiting setup using SlowAPI.

Uses in-memory storage (default) so it works without Redis.
Redis-backed storage can be enabled in production by passing a
storage_uri to Limiter, but it is NOT required for development.
"""

from __future__ import annotations

from fastapi import Request
from slowapi import Limiter
from slowapi.util import get_remote_address

# In-memory rate limiter — no Redis dependency required.
# For production with Redis: Limiter(key_func=..., storage_uri="redis://localhost:6379")
limiter = Limiter(key_func=get_remote_address, default_limits=[])


def rate_limit_custom(request: Request) -> str:
    """Helper if we decide to key-limit on user ID claims rather than IP."""
    claims = getattr(request.state, "claims", None)
    if claims and "sub" in claims:
        return claims["sub"]
    return get_remote_address(request)
