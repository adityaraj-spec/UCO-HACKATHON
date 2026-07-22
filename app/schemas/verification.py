"""
app/schemas/verification.py

Pydantic schemas for speaker verification requests, results, and history.
"""

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class VerificationRequestMeta(BaseModel):
    """
    Optional metadata accompanying a verification request.

    `layer1_score` represents the AI-voice-detection probability produced by
    Layer 1 (0.0 = certainly human, 1.0 = certainly AI-generated). If Layer 1
    has not been run yet (e.g. Layer 2 is being tested in isolation), this
    defaults to 0.0, meaning "Layer 1 passed cleanly".
    """

    layer1_score: float = Field(default=0.0, ge=0.0, le=1.0)


class VerificationResult(BaseModel):
    """Response payload returned by POST /api/v1/verify."""

    user_id: uuid.UUID
    similarity_score: float
    verified: bool
    decision: str  # "verified" | "step_up" | "mismatch" | "REPLAY_DETECTED" | transaction decisions

    # Risk engine output (Layer 1 + raw Layer 2 similarity combined)
    layer1_score: float
    risk_score: float
    risk_level: str  # "CLEAN" | "FRAUD_ALERT"
    stage_timings_ms: dict[str, float] | None = None

    # Optional transaction-aware controls. For legacy identity-only requests,
    # these remain None/False and existing callers can ignore them.
    transaction_type: str | None = None
    transaction_amount: float | None = None
    transaction_tier: str | None = None
    phrase_match: bool | None = None
    biometric_verified: bool | None = None
    otp_required: bool = False
    final_authorized: bool | None = None


class VerificationLogRead(BaseModel):
    """A single historical verification record."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    user_id: uuid.UUID
    similarity_score: float
    decision: str
    created_at: datetime


class SpeakerMatch(BaseModel):
    user_id: uuid.UUID
    similarity_score: float


class IdentificationResult(BaseModel):
    matches: list[SpeakerMatch]
