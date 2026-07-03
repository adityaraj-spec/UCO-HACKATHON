"""
streamlit_app/api_client.py

Thin HTTP client wrapping calls to the PhaseGuard Layer 2 FastAPI backend.

Every function returns (success: bool, data_or_error).
On success, data_or_error is the parsed JSON. On failure it is a string.
"""

from typing import Any

import requests

from config import API_V1

DEFAULT_TIMEOUT = 30
ENROLL_TIMEOUT = 60


# ------------------------------------------------------------------ #
# Helpers                                                             #
# ------------------------------------------------------------------ #

def _handle_response(response: requests.Response) -> tuple[bool, Any]:
    """Convert a requests.Response into (success, payload_or_error_message)."""
    try:
        payload = response.json()
    except ValueError:
        payload = {"detail": response.text or "Empty response from server"}

    if response.ok:
        return True, payload

    detail = payload.get("detail", f"Request failed with status {response.status_code}")
    if isinstance(detail, list):
        detail = "; ".join(str(item.get("msg", item)) for item in detail)
    return False, str(detail)


# ------------------------------------------------------------------ #
# User Management                                                     #
# ------------------------------------------------------------------ #

def create_user(name: str, email: str) -> tuple[bool, Any]:
    """POST /api/v1/users"""
    try:
        response = requests.post(
            f"{API_V1}/users",
            json={"name": name, "email": email},
            timeout=DEFAULT_TIMEOUT,
        )
    except requests.RequestException as exc:
        return False, f"Could not reach API: {exc}"
    return _handle_response(response)


def get_user(user_id: str) -> tuple[bool, Any]:
    """GET /api/v1/users/{id}"""
    try:
        response = requests.get(f"{API_V1}/users/{user_id}", timeout=DEFAULT_TIMEOUT)
    except requests.RequestException as exc:
        return False, f"Could not reach API: {exc}"
    return _handle_response(response)


# ------------------------------------------------------------------ #
# Consent                                                             #
# ------------------------------------------------------------------ #

def grant_consent(user_id: str, consent_type: str = "EXPLICIT_OPT_IN") -> tuple[bool, Any]:
    """POST /consent/grant — grant DPDP biometric consent for a user."""
    try:
        response = requests.post(
            f"{API_V1}/consent/grant",
            data={"user_id": user_id, "consent_type": consent_type},
            timeout=DEFAULT_TIMEOUT,
        )
    except requests.RequestException as exc:
        return False, f"Could not reach API: {exc}"
    return _handle_response(response)


def get_consent_status(user_id: str) -> tuple[bool, Any]:
    """GET /consent/status/{user_id}"""
    try:
        response = requests.get(
            f"{API_V1}/consent/status/{user_id}",
            timeout=DEFAULT_TIMEOUT,
        )
    except requests.RequestException as exc:
        return False, f"Could not reach API: {exc}"
    return _handle_response(response)


# ------------------------------------------------------------------ #
# Session-based Enrollment (new backend flow)                         #
# ------------------------------------------------------------------ #

def create_enrollment_session(user_id: str) -> tuple[bool, Any]:
    """POST /voice/enroll/session — create a new 5-sample enrollment session."""
    try:
        response = requests.post(
            f"{API_V1}/voice/enroll/session",
            data={"user_id": user_id},
            timeout=DEFAULT_TIMEOUT,
        )
    except requests.RequestException as exc:
        return False, f"Could not reach API: {exc}"
    return _handle_response(response)


def submit_enrollment_sample(
    user_id: str,
    session_id: str,
    filename: str,
    file_bytes: bytes,
    mime_type: str,
) -> tuple[bool, Any]:
    """POST /voice/enroll/sample — submit one audio sample for enrollment."""
    try:
        response = requests.post(
            f"{API_V1}/voice/enroll/sample",
            data={"user_id": user_id, "session_id": session_id},
            files={"audio": (filename, file_bytes, mime_type)},
            timeout=ENROLL_TIMEOUT,
        )
    except requests.RequestException as exc:
        return False, f"Could not reach API: {exc}"
    return _handle_response(response)


def complete_enrollment(user_id: str, session_id: str) -> tuple[bool, Any]:
    """POST /voice/enroll/complete — finalize enrollment session."""
    try:
        response = requests.post(
            f"{API_V1}/voice/enroll/complete",
            data={"user_id": user_id, "session_id": session_id},
            timeout=ENROLL_TIMEOUT,
        )
    except requests.RequestException as exc:
        return False, f"Could not reach API: {exc}"
    return _handle_response(response)


# ------------------------------------------------------------------ #
# Challenge-Response Verification (new backend flow)                  #
# ------------------------------------------------------------------ #

def get_challenge(user_id: str, session_id: str) -> tuple[bool, Any]:
    """GET /voice/challenge — get a spoken challenge phrase + JWT token."""
    try:
        response = requests.get(
            f"{API_V1}/voice/challenge",
            params={"user_id": user_id, "session_id": session_id},
            timeout=DEFAULT_TIMEOUT,
        )
    except requests.RequestException as exc:
        return False, f"Could not reach API: {exc}"
    return _handle_response(response)


def authenticate_voice(
    user_id: str,
    session_id: str,
    challenge_token: str,
    device_fingerprint: str,
    filename: str,
    file_bytes: bytes,
    mime_type: str,
) -> tuple[bool, Any]:
    """POST /voice/authenticate — verify voice against enrolled voiceprint."""
    try:
        response = requests.post(
            f"{API_V1}/voice/authenticate",
            data={
                "user_id": user_id,
                "session_id": session_id,
                "challenge_token": challenge_token,
                "device_fingerprint": device_fingerprint,
            },
            files={"audio": (filename, file_bytes, mime_type)},
            timeout=DEFAULT_TIMEOUT,
        )
    except requests.RequestException as exc:
        return False, f"Could not reach API: {exc}"
    return _handle_response(response)


# ------------------------------------------------------------------ #
# Legacy fallback (kept for History / Risk pages)                     #
# ------------------------------------------------------------------ #

def enroll_user(user_id: str, files: list[tuple[str, bytes, str]]) -> tuple[bool, Any]:
    """POST /api/v1/enroll (legacy single-call)"""
    try:
        multipart_files = [
            ("files", (filename, content, mime_type))
            for filename, content, mime_type in files
        ]
        response = requests.post(
            f"{API_V1}/enroll",
            data={"user_id": user_id},
            files=multipart_files,
            timeout=120,
        )
    except requests.RequestException as exc:
        return False, f"Could not reach API: {exc}"
    return _handle_response(response)


def verify_user(
    user_id: str,
    filename: str,
    file_bytes: bytes,
    mime_type: str,
    layer1_score: float = 0.0,
) -> tuple[bool, Any]:
    """POST /api/v1/verify (legacy single-call)"""
    try:
        response = requests.post(
            f"{API_V1}/verify",
            data={"user_id": user_id, "layer1_score": str(layer1_score)},
            files={"file": (filename, file_bytes, mime_type)},
            timeout=DEFAULT_TIMEOUT,
        )
    except requests.RequestException as exc:
        return False, f"Could not reach API: {exc}"
    return _handle_response(response)


def get_verification_history(user_id: str, limit: int = 50) -> tuple[bool, Any]:
    """GET /api/v1/verification-history/{id}"""
    try:
        response = requests.get(
            f"{API_V1}/verification-history/{user_id}",
            params={"limit": limit},
            timeout=DEFAULT_TIMEOUT,
        )
    except requests.RequestException as exc:
        return False, f"Could not reach API: {exc}"
    return _handle_response(response)


def get_risk_history(user_id: str, limit: int = 50) -> tuple[bool, Any]:
    """GET /api/v1/risk-history/{id}"""
    try:
        response = requests.get(
            f"{API_V1}/risk-history/{user_id}",
            params={"limit": limit},
            timeout=DEFAULT_TIMEOUT,
        )
    except requests.RequestException as exc:
        return False, f"Could not reach API: {exc}"
    return _handle_response(response)


def check_health() -> tuple[bool, Any]:
    """GET /health"""
    try:
        response = requests.get(
            f"{API_V1.rsplit('/api/v1', 1)[0]}/health", timeout=10
        )
    except requests.RequestException as exc:
        return False, f"Could not reach API: {exc}"
    return _handle_response(response)
