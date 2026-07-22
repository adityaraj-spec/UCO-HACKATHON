"""
app/api/v1/router.py

Aggregates all v1 endpoint routers into a single APIRouter, mounted under
the "/api/v1" prefix in app/main.py.
"""

from fastapi import APIRouter

from app.api.v1.endpoints import admin, challenge, enroll, history, kyc, users, verify

api_router = APIRouter()

api_router.include_router(enroll.router, tags=["Enrollment"])
api_router.include_router(verify.router, tags=["Verification"])
api_router.include_router(users.router, tags=["Users"])
api_router.include_router(history.router, tags=["History"])
api_router.include_router(challenge.router, tags=["Challenge Liveness"])
api_router.include_router(kyc.router, tags=["Voice KYC"])
api_router.include_router(admin.router, tags=["Admin & Compliance"])
from app.api.v1.endpoints import stream
api_router.include_router(stream.router, tags=['Streaming'])
