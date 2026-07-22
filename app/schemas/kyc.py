"""Pydantic schemas for dynamic Voice KYC enrollment."""

import uuid
from datetime import datetime

from pydantic import BaseModel, Field


class KYCSessionStartRequest(BaseModel):
    phone_number: str = Field(min_length=6, max_length=20)


class KYCSessionStartResponse(BaseModel):
    session_id: uuid.UUID


class KYCChallengeRequest(BaseModel):
    session_id: uuid.UUID
    attempt_no: int = Field(ge=1, le=2)


class KYCChallengeResponse(BaseModel):
    sentence_text: str
    expires_at: datetime


class KYCAttemptResponse(BaseModel):
    success: bool
    asr_match_score: float
    message: str


class KYCEnrolRequest(BaseModel):
    session_id: uuid.UUID


class KYCEnrolResponse(BaseModel):
    success: bool
    salt_id: str
    message: str
