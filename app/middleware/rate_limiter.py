"""
app/middleware/rate_limiter.py

FastAPI rate limiting setup using SlowAPI.

Limits:
  - Enrollment endpoints (high overhead, Speechbrain load): 3/hour
  - Verification/Auth endpoints: 10/minute
  - Health/General checks: 100/minute
"""

from __future__ import annotations

from fastapi import Request
from slowapi import Limiter
from slowapi.util import get_remote_address

# Configure rate limiter using client IP as key finder
limiter = Limiter(key_func=get_remote_address)


def rate_limit_custom(request: Request) -> str:
    """Helper if we decide to key-limit on user ID claims rather than IP."""
    claims = getattr(request.state, "claims", None)
    if claims and "sub" in claims:
        return claims["sub"]
    return get_remote_address(request)
