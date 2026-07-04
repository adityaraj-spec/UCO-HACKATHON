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


def _auth_headers(token: str | None = None) -> dict[str, str]:
    if not token:
        try:
            import streamlit as st

            token = st.session_state.get("session_token")
        except Exception:
            token = None
    return {"Authorization": f"Bearer {token}"} if token else {}


# ------------------------------------------------------------------ #
# Banking Auth                                                        #
# ------------------------------------------------------------------ #

def register_customer(payload: dict[str, Any]) -> tuple[bool, Any]:
    """POST /api/v1/auth/register"""
    try:
        response = requests.post(
            f"{API_V1}/auth/register",
            json=payload,
            timeout=DEFAULT_TIMEOUT,
        )
    except requests.RequestException as exc:
        return False, f"Could not reach API: {exc}"
    return _handle_response(response)


def login(account_number: str, mpin: str, device_fingerprint: str | None = None) -> tuple[bool, Any]:
    """POST /api/v1/auth/login"""
    import uuid
    from datetime import datetime, timezone

    payload = {
        "account_number": account_number,
        "mpin": mpin,
        "request_nonce": str(uuid.uuid4()),
        "request_timestamp": datetime.now(timezone.utc).isoformat(),
        "device_fingerprint": device_fingerprint,
    }
    try:
        response = requests.post(f"{API_V1}/auth/login", json=payload, timeout=DEFAULT_TIMEOUT)
    except requests.RequestException as exc:
        return False, f"Could not reach API: {exc}"
    return _handle_response(response)


def get_me() -> tuple[bool, Any]:
    """GET /api/v1/auth/me"""
    try:
        response = requests.get(f"{API_V1}/auth/me", headers=_auth_headers(), timeout=DEFAULT_TIMEOUT)
    except requests.RequestException as exc:
        return False, f"Could not reach API: {exc}"
    return _handle_response(response)


def change_mpin(old_mpin: str, new_mpin: str) -> tuple[bool, Any]:
    try:
        response = requests.post(
            f"{API_V1}/auth/mpin/change",
            json={"old_mpin": old_mpin, "new_mpin": new_mpin},
            headers=_auth_headers(),
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
            data={"consent_type": consent_type},
            headers=_auth_headers(),
            timeout=DEFAULT_TIMEOUT,
        )
    except requests.RequestException as exc:
        return False, f"Could not reach API: {exc}"
    return _handle_response(response)


def get_consent_status(user_id: str | None = None) -> tuple[bool, Any]:
    """GET /consent/status/{user_id}"""
    try:
        response = requests.get(
            f"{API_V1}/consent/status",
            headers=_auth_headers(),
            timeout=DEFAULT_TIMEOUT,
        )
    except requests.RequestException as exc:
        return False, f"Could not reach API: {exc}"
    return _handle_response(response)


# ------------------------------------------------------------------ #
# Session-based Enrollment (new backend flow)                         #
# ------------------------------------------------------------------ #

def create_enrollment_session(user_id: str | None = None) -> tuple[bool, Any]:
    """POST /voice/enroll/session — create a new 5-sample enrollment session."""
    try:
        response = requests.post(
            f"{API_V1}/voice/enroll/session",
            headers=_auth_headers(),
            timeout=DEFAULT_TIMEOUT,
        )
    except requests.RequestException as exc:
        return False, f"Could not reach API: {exc}"
    return _handle_response(response)


def submit_enrollment_sample(
    user_id: str | None,
    session_id: str,
    filename: str,
    file_bytes: bytes,
    mime_type: str,
) -> tuple[bool, Any]:
    """POST /voice/enroll/sample — submit one audio sample for enrollment."""
    try:
        response = requests.post(
            f"{API_V1}/voice/enroll/sample",
            data={"session_id": session_id},
            files={"audio": (filename, file_bytes, mime_type)},
            headers=_auth_headers(),
            timeout=ENROLL_TIMEOUT,
        )
    except requests.RequestException as exc:
        return False, f"Could not reach API: {exc}"
    return _handle_response(response)


def complete_enrollment(user_id: str | None, session_id: str) -> tuple[bool, Any]:
    """POST /voice/enroll/complete — finalize enrollment session."""
    try:
        response = requests.post(
            f"{API_V1}/voice/enroll/complete",
            data={"session_id": session_id},
            headers=_auth_headers(),
            timeout=ENROLL_TIMEOUT,
        )
    except requests.RequestException as exc:
        return False, f"Could not reach API: {exc}"
    return _handle_response(response)


# ------------------------------------------------------------------ #
# Challenge-Response Verification (new backend flow)                  #
# ------------------------------------------------------------------ #

def get_challenge(user_id: str | None, session_id: str) -> tuple[bool, Any]:
    """GET /voice/challenge — get a spoken challenge phrase + JWT token."""
    try:
        response = requests.get(
            f"{API_V1}/voice/challenge",
            params={"session_id": session_id},
            headers=_auth_headers(),
            timeout=DEFAULT_TIMEOUT,
        )
    except requests.RequestException as exc:
        return False, f"Could not reach API: {exc}"
    return _handle_response(response)


def authenticate_voice(
    user_id: str | None,
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
                "session_id": session_id,
                "challenge_token": challenge_token,
                "device_fingerprint": device_fingerprint,
            },
            files={"audio": (filename, file_bytes, mime_type)},
            headers=_auth_headers(),
            timeout=DEFAULT_TIMEOUT,
        )
    except requests.RequestException as exc:
        return False, f"Could not reach API: {exc}"
    return _handle_response(response)


def authenticate_voice_direct(
    user_id: str | None,
    device_fingerprint: str,
    filename: str,
    file_bytes: bytes,
    mime_type: str,
) -> tuple[bool, Any]:
    """POST /voice/authenticate/direct — verify voice directly against enrolled voiceprint."""
    try:
        response = requests.post(
            f"{API_V1}/voice/authenticate/direct",
            data={"device_fingerprint": device_fingerprint},
            files={"audio": (filename, file_bytes, mime_type)},
            headers=_auth_headers(),
            timeout=DEFAULT_TIMEOUT,
        )
    except requests.RequestException as exc:
        return False, f"Could not reach API: {exc}"
    return _handle_response(response)



# ------------------------------------------------------------------ #
# Legacy fallback (kept for History / Risk pages)                     #
# ------------------------------------------------------------------ #

def enroll_user(user_id: str | None, files: list[tuple[str, bytes, str]]) -> tuple[bool, Any]:
    """POST /api/v1/enroll (legacy single-call)"""
    try:
        multipart_files = [
            ("files", (filename, content, mime_type))
            for filename, content, mime_type in files
        ]
        response = requests.post(
            f"{API_V1}/enroll",
            files=multipart_files,
            headers=_auth_headers(),
            timeout=120,
        )
    except requests.RequestException as exc:
        return False, f"Could not reach API: {exc}"
    return _handle_response(response)


def verify_user(
    user_id: str | None,
    filename: str,
    file_bytes: bytes,
    mime_type: str,
    layer1_score: float = 0.0,
) -> tuple[bool, Any]:
    """POST /api/v1/verify (legacy single-call)"""
    try:
        response = requests.post(
            f"{API_V1}/verify",
            data={"layer1_score": str(layer1_score)},
            files={"file": (filename, file_bytes, mime_type)},
            headers=_auth_headers(),
            timeout=DEFAULT_TIMEOUT,
        )
    except requests.RequestException as exc:
        return False, f"Could not reach API: {exc}"
    return _handle_response(response)


def get_verification_history(user_id: str | None = None, limit: int = 50) -> tuple[bool, Any]:
    """GET /api/v1/verification-history/me"""
    try:
        response = requests.get(
            f"{API_V1}/verification-history/me",
            params={"limit": limit},
            headers=_auth_headers(),
            timeout=DEFAULT_TIMEOUT,
        )
    except requests.RequestException as exc:
        return False, f"Could not reach API: {exc}"
    return _handle_response(response)


def get_risk_history(user_id: str | None = None, limit: int = 50) -> tuple[bool, Any]:
    """GET /api/v1/risk-history/me"""
    try:
        response = requests.get(
            f"{API_V1}/risk-history/me",
            params={"limit": limit},
            headers=_auth_headers(),
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
