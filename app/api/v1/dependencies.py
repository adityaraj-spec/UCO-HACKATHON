"""
app/api/v1/dependencies.py

FastAPI dependencies for rate limiting and JWT replay protection.
"""

from fastapi import Request, HTTPException, status, Header
from app.services.redis_service import redis_service
from app.core.logging import get_logger

log = get_logger(__name__)

async def rate_limiter(request: Request):
    """
    Dependency that enforces rate limits on incoming API requests.
    Enforces a limit of 10 requests per minute per IP address.
    """
    # Try to get client IP, respect X-Forwarded-For if behind a proxy
    ip = request.headers.get("x-forwarded-for")
    if ip:
        ip = ip.split(",")[0].strip()
    else:
        ip = request.client.host if request.client else "unknown"
        
    path = request.url.path
    # Rate limit key combines IP and path to rate limit specific endpoints
    key = f"{ip}:{path}"
    
    # 10 requests per 60 seconds
    limit = 10
    window = 60
    
    if redis_service.is_rate_limited(key, limit=limit, window_seconds=window):
        log.warning(f"Rate limit hit for key={key} on {path}")
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many requests. Please try again in a minute."
        )

async def replay_protection(x_jti: str = Header(None, alias="X-JTI")):
    """
    Dependency that checks the X-JTI header to prevent request replay attacks.
    If X-JTI is provided, it marks it as used for 10 minutes (600 seconds).
    If the same X-JTI is sent again within that period, the request is rejected.
    """
    if x_jti:
        # Check and mark token as used for 10 minutes
        is_fresh = redis_service.mark_token_used(x_jti, ttl_seconds=600)
        if not is_fresh:
            log.warning(f"Replay attack detected for JTI={x_jti}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Request replay detected. Token already used."
            )
