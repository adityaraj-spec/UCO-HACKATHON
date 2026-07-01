"""
app/middleware/rbac.py

FastAPI Role-Based Access Control (RBAC) dependencies.
"""

from __future__ import annotations

from fastapi import Depends, HTTPException, status
from app.middleware.jwt_auth import get_current_user_claims


class RoleChecker:
    """Dependency helper to enforce authorization roles on endpoints."""

    def __init__(self, allowed_roles: list[str]) -> None:
        self.allowed_roles = allowed_roles

    def __call__(self, claims: dict = Depends(get_current_user_claims)) -> dict:
        user_role = claims.get("role", "CUSTOMER")
        if user_role not in self.allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Required role missing. Allowed: {self.allowed_roles}"
            )
        return claims


# Predefined helpers
allow_customer = RoleChecker(["CUSTOMER", "AGENT", "ADMIN", "SERVICE"])
allow_agent_or_admin = RoleChecker(["AGENT", "ADMIN"])
allow_admin_only = RoleChecker(["ADMIN"])
