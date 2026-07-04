"""
app/schemas/user.py

Pydantic schemas for banking identity and public customer profiles.
"""

from __future__ import annotations

import uuid
from datetime import date, datetime

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator


class BankCustomerCreate(BaseModel):
    """Bank-staff payload for provisioning a customer account."""

    account_number: str = Field(..., min_length=6, max_length=20)
    mpin: str = Field(..., min_length=6, max_length=6)
    full_name: str = Field(..., min_length=1, max_length=255)
    phone_number: str = Field(..., min_length=10, max_length=20)
    date_of_birth: date
    branch_code: str = Field(..., min_length=2, max_length=10)
    account_type: str = "SAVINGS"
    email: EmailStr | None = None

    @field_validator("mpin")
    @classmethod
    def mpin_must_be_six_digits(cls, value: str) -> str:
        if not value.isdigit():
            raise ValueError("MPIN must contain exactly 6 digits.")
        return value

    @field_validator("account_type")
    @classmethod
    def normalize_account_type(cls, value: str) -> str:
        normalized = value.upper()
        if normalized not in {"SAVINGS", "CURRENT", "NRE"}:
            raise ValueError("account_type must be SAVINGS, CURRENT, or NRE.")
        return normalized


class MPINLoginRequest(BaseModel):
    """Customer login payload using bank-visible credentials."""

    account_number: str = Field(..., min_length=6, max_length=20)
    mpin: str = Field(..., min_length=6, max_length=6)
    request_nonce: str | None = Field(None, max_length=128)
    request_timestamp: datetime | None = None
    device_fingerprint: str | None = Field(None, max_length=512)


class MPINChangeRequest(BaseModel):
    old_mpin: str = Field(..., min_length=6, max_length=6)
    new_mpin: str = Field(..., min_length=6, max_length=6)

    @field_validator("old_mpin", "new_mpin")
    @classmethod
    def mpin_must_be_six_digits(cls, value: str) -> str:
        if not value.isdigit():
            raise ValueError("MPIN must contain exactly 6 digits.")
        return value


class AccountNumberRequest(BaseModel):
    account_number: str = Field(..., min_length=6, max_length=20)


class MPINLoginResponse(BaseModel):
    success: bool
    user_id: uuid.UUID
    account_number: str
    full_name: str
    session_token: str
    is_enrolled: bool
    kyc_status: str
    mpin_rotation_required: bool = False
    new_device_login: bool = False


class UserPublicRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    account_number: str
    full_name: str
    phone_number: str
    branch_code: str | None
    account_type: str
    kyc_status: str
    is_enrolled: bool
    last_login_at: datetime | None


class UserRead(BaseModel):
    """Compatibility response for legacy developer-facing user lookups."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    account_number: str
    full_name: str
    email: EmailStr | None = None
    created_at: datetime


class UserDetailRead(UserRead):
    is_enrolled: bool
    recording_count: int = 0
    embedding_dimension: int | None = None
    voiceprint_updated_at: datetime | None = None


def mask_phone(phone_number: str) -> str:
    """Return a customer-safe phone representation, e.g. XXXXXX7890."""
    return f"{'X' * max(len(phone_number) - 4, 0)}{phone_number[-4:]}"
