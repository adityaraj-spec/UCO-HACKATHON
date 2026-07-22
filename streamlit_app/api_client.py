"""
streamlit_app/api_client.py

Thin HTTP client wrapping calls to the PhaseGuard Layer 2 FastAPI backend.

Every function returns a tuple of (success: bool, data_or_error). On
success, `data_or_error` is the parsed JSON response. On failure, it is a
human-readable error message extracted from the API's error response (or a
generic message if the API was unreachable).

Keeping all HTTP calls in one module makes it trivial to swap the backend
URL, add auth headers, or add retry logic later without touching every page.
"""

from typing import Any

import requests

from config import API_V1

DEFAULT_TIMEOUT = 120  # seconds; audio loading, embedding extraction, & cloud DB logging can take extra time


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
        # FastAPI validation errors are returned as a list of dicts.
        detail = "; ".join(str(item.get("msg", item)) for item in detail)
    return False, str(detail)


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


def enroll_user(
    user_id: str,
    files: list[tuple[str, bytes, str]],
    channel: str = "DIRECT_API",
    biometric_consent_confirmed: bool = False,
    identity_confirmed: bool = False,
    authenticated: bool = False,
    otp_verified: bool = False,
    device_id: str | None = None,
    branch_officer_id: str | None = None,
) -> tuple[bool, Any]:
    """
    POST /api/v1/enroll

    Args:
        user_id: UUID of the user to enroll.
        files: list of (filename, file_bytes, mime_type) tuples for each
               recording.
    """
    try:
        multipart_files = [
            ("files", (filename, content, mime_type))
            for filename, content, mime_type in files
        ]
        form_data: dict[str, str] = {
            "user_id": user_id,
            "channel": channel,
            "biometric_consent_confirmed": str(biometric_consent_confirmed).lower(),
            "identity_confirmed": str(identity_confirmed).lower(),
            "authenticated": str(authenticated).lower(),
            "otp_verified": str(otp_verified).lower(),
        }
        if device_id:
            form_data["device_id"] = device_id
        if branch_officer_id:
            form_data["branch_officer_id"] = branch_officer_id

        response = requests.post(
            f"{API_V1}/enroll",
            data=form_data,
            files=multipart_files,
            timeout=120,  # enrollment processes many files; allow extra time
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
    transaction_type: str | None = None,
    transaction_amount: float | None = None,
    expected_phrase: str | None = None,
    spoken_text: str | None = None,
) -> tuple[bool, Any]:
    """POST /api/v1/verify"""
    try:
        form_data: dict[str, str] = {
            "user_id": user_id,
            "layer1_score": str(layer1_score),
        }
        if transaction_type:
            form_data["transaction_type"] = transaction_type
        if transaction_amount is not None:
            form_data["transaction_amount"] = str(transaction_amount)
        if expected_phrase:
            form_data["expected_phrase"] = expected_phrase
        if spoken_text:
            form_data["spoken_text"] = spoken_text

        response = requests.post(
            f"{API_V1}/verify",
            data=form_data,
            files={"file": (filename, file_bytes, mime_type)},
            timeout=DEFAULT_TIMEOUT,
        )
    except requests.RequestException as exc:
        return False, f"Could not reach API: {exc}"
    return _handle_response(response)


def identify_speaker(
    filename: str,
    file_bytes: bytes,
    mime_type: str,
    k: int = 5,
) -> tuple[bool, Any]:
    """POST /api/v1/identify"""
    try:
        response = requests.post(
            f"{API_V1}/identify",
            data={"k": str(k)},
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


def start_kyc_session(phone_number: str) -> tuple[bool, Any]:
    """POST /api/v1/kyc/session/start"""
    try:
        response = requests.post(
            f"{API_V1}/kyc/session/start",
            json={"phone_number": phone_number},
            timeout=DEFAULT_TIMEOUT,
        )
    except requests.RequestException as exc:
        return False, f"Could not reach API: {exc}"
    return _handle_response(response)


def get_challenge_sentence(session_id: str, attempt_no: int) -> tuple[bool, Any]:
    """POST /api/v1/kyc/challenge"""
    try:
        response = requests.post(
            f"{API_V1}/kyc/challenge",
            json={"session_id": session_id, "attempt_no": attempt_no},
            timeout=DEFAULT_TIMEOUT,
        )
    except requests.RequestException as exc:
        return False, f"Could not reach API: {exc}"
    return _handle_response(response)


def submit_attempt(
    session_id: str,
    attempt_no: int,
    audio_filename: str,
    audio_bytes: bytes,
    mime_type: str,
    asr_transcript: str | None = None,
) -> tuple[bool, Any]:
    """POST /api/v1/kyc/attempt"""
    try:
        data = {"session_id": session_id, "attempt_no": str(attempt_no)}
        if asr_transcript:
            data["asr_transcript"] = asr_transcript
        response = requests.post(
            f"{API_V1}/kyc/attempt",
            data=data,
            files={"audio": (audio_filename, audio_bytes, mime_type)},
            timeout=DEFAULT_TIMEOUT,
        )
    except requests.RequestException as exc:
        return False, f"Could not reach API: {exc}"
    return _handle_response(response)


def enrol_voice(session_id: str) -> tuple[bool, Any]:
    """POST /api/v1/kyc/enrol"""
    try:
        response = requests.post(
            f"{API_V1}/kyc/enrol",
            json={"session_id": session_id},
            timeout=DEFAULT_TIMEOUT,
        )
    except requests.RequestException as exc:
        return False, f"Could not reach API: {exc}"
    return _handle_response(response)
