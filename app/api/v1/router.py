"""
app/api/v1/router.py

Aggregates all v1 endpoint routers into a single APIRouter, mounted under
the "/api/v1" prefix in app/main.py.
"""

from fastapi import APIRouter

from app.api.v1.endpoints import enroll, history, users, verify

api_router = APIRouter()

api_router.include_router(enroll.router, tags=["Enrollment"])
api_router.include_router(verify.router, tags=["Verification"])
api_router.include_router(users.router, tags=["Users"])
api_router.include_router(history.router, tags=["History"])
from app.api.v1.endpoints import stream
api_router.include_router(stream.router, tags=['Streaming'])
