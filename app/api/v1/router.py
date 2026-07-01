"""
app/api/v1/router.py

Aggregates all v1 endpoint routers into a single APIRouter, mounted under
the "/api/v1" prefix in app/main.py.
"""

from fastapi import APIRouter

from app.api.v1.endpoints import enroll, history, users, verify, voice_auth

api_router = APIRouter()

# Layer 2 Enhanced RBI endpoints
api_router.include_router(voice_auth.router)

# Layer 1 Backwards compatible/fallback endpoints
api_router.include_router(enroll.router, tags=["Enrollment"])
api_router.include_router(verify.router, tags=["Verification"])
api_router.include_router(users.router, tags=["Users"])
api_router.include_router(history.router, tags=["History"])

