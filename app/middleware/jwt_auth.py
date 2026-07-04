"""
app/middleware/jwt_auth.py

FastAPI dependency for verifying JWT OAuth2 tokens and extracting user claims.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timedelta, timezone
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt

from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()
security = HTTPBearer()


def create_access_token(
    *,
    user_id: uuid.UUID,
    account_number: str,
    full_name: str,
    role: str = "CUSTOMER",
    expires_delta: timedelta | None = None,
) -> str:
    """Issue a short-lived JWT with banking identity claims."""
    now = datetime.now(timezone.utc)
    expires = now + (
        expires_delta
        if expires_delta is not None
        else timedelta(minutes=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    payload = {
        "sub": str(user_id),
        "account_number": account_number,
        "full_name": full_name,
        "role": role,
        "iat": int(now.timestamp()),
        "exp": int(expires.timestamp()),
        "jti": uuid.uuid4().hex,
    }
    return jwt.encode(payload, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)


def get_current_user_claims(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> dict:
    """
    Extract claims from JSON Web Token. Throws 401 if invalid.
    """
    token = credentials.credentials
    try:
        payload = jwt.decode(
            token,
            settings.JWT_SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM]
        )
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication token signature has expired.",
            headers={"WWW-Authenticate": "Bearer"},
        )


def get_current_user_id(claims: dict = Depends(get_current_user_claims)) -> uuid.UUID:
    """Resolve the internal UUID from JWT claims."""
    try:
        return uuid.UUID(claims["sub"])
    except (KeyError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication token missing a valid subject.",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc
    except jwt.PyJWTError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid authentication token: {e}",
            headers={"WWW-Authenticate": "Bearer"},
        )
