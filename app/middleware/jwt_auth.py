"""
app/middleware/jwt_auth.py

FastAPI dependency for verifying JWT OAuth2 tokens and extracting user claims.
"""

from __future__ import annotations

import logging
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt

from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()
security = HTTPBearer()


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
            algorithms=["HS256"]
        )
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication token signature has expired.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.PyJWTError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid authentication token: {e}",
            headers={"WWW-Authenticate": "Bearer"},
        )
