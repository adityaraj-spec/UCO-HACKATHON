"""
Dynamic Voice KYC enrolment page.

The flow uses two fresh spoken sentence challenges. Each sentence is issued
server-side, expires quickly, and can be used only once.
"""

import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

import api_client

st.set_page_config(page_title="PhaseGuard - Voice KYC", layout="wide")


def _state(key: str, default=None):
    if key not in st.session_state:
        st.session_state[key] = default
    return st.session_state[key]


def _reset_session() -> None:
    for key in [
        "kyc_session_id",
        "kyc_phone_number",
        "kyc_challenge_1",
        "kyc_challenge_2",
        "kyc_attempt_1_ok",
        "kyc_attempt_2_ok",
        "kyc_enrolment_done",
        "kyc_audio_nonce_1",
        "kyc_audio_nonce_2",
    ]:
        st.session_state.pop(key, None)


def _remaining_seconds(expires_at: str | None) -> int:
    if not expires_at:
        return 0
    value = expires_at.replace("Z", "+00:00")
    expires = datetime.fromisoformat(value)
    if expires.tzinfo is None:
        expires = expires.replace(tzinfo=timezone.utc)
    return max(0, int((expires - datetime.now(timezone.utc)).total_seconds()))


def _audio_capture(attempt_no: int):
    nonce = st.session_state.get(f"kyc_audio_nonce_{attempt_no}", 0)
    if hasattr(st, "audio_input"):
        return st.audio_input("Record the sentence", key=f"kyc_audio_{attempt_no}_{nonce}")
    return st.file_uploader(
        "Upload a recording of the sentence",
        type=["wav", "mp3", "m4a", "ogg", "flac"],
        key=f"kyc_audio_upload_{attempt_no}_{nonce}",
    )


def _attempt_panel(attempt_no: int) -> None:
    session_id = _state("kyc_session_id")
    challenge_key = f"kyc_challenge_{attempt_no}"
    attempt_ok_key = f"kyc_attempt_{attempt_no}_ok"
    challenge = _state(challenge_key)
    attempt_ok = _state(attempt_ok_key, False)

    st.subheader(f"Attempt {attempt_no}")

    if attempt_ok:
        st.success(f"Attempt {attempt_no} completed successfully!")
        if st.button(f"Reattempt Attempt {attempt_no}", key=f"kyc_reattempt_{attempt_no}"):
            st.session_state[attempt_ok_key] = False
            st.session_state[challenge_key] = None
            st.session_state[f"kyc_audio_nonce_{attempt_no}"] = (
                st.session_state.get(f"kyc_audio_nonce_{attempt_no}", 0) + 1
            )
            st.rerun()
        return

    cols = st.columns([1, 1, 4])
    with cols[0]:
        if st.button("Get sentence", key=f"kyc_get_sentence_{attempt_no}"):
            ok, payload = api_client.get_challenge_sentence(session_id, attempt_no)
            if ok:
                st.session_state[challenge_key] = payload
                st.session_state[attempt_ok_key] = False
                st.session_state[f"kyc_audio_nonce_{attempt_no}"] = (
                    st.session_state.get(f"kyc_audio_nonce_{attempt_no}", 0) + 1
                )
                st.rerun()
            else:
                st.error(payload)
    with cols[1]:
        if challenge and st.button("New sentence", key=f"kyc_new_sentence_{attempt_no}"):
            ok, payload = api_client.get_challenge_sentence(session_id, attempt_no)
            if ok:
                st.session_state[challenge_key] = payload
                st.session_state[attempt_ok_key] = False
                st.session_state[f"kyc_audio_nonce_{attempt_no}"] = (
                    st.session_state.get(f"kyc_audio_nonce_{attempt_no}", 0) + 1
                )
                st.rerun()
            else:
                st.error(payload)

    if not challenge:
        st.info("Request a fresh sentence to begin this attempt.")
        return

    remaining = _remaining_seconds(challenge.get("expires_at"))
    if remaining <= 0:
        st.error("This sentence has expired.")
        if st.button(f"Reattempt with fresh sentence", key=f"kyc_expired_reattempt_{attempt_no}"):
            ok, payload = api_client.get_challenge_sentence(session_id, attempt_no)
            if ok:
                st.session_state[challenge_key] = payload
                st.session_state[attempt_ok_key] = False
                st.session_state[f"kyc_audio_nonce_{attempt_no}"] = (
                    st.session_state.get(f"kyc_audio_nonce_{attempt_no}", 0) + 1
                )
                st.rerun()
            else:
                st.error(payload)
        return

    st.markdown("Please read this banking transaction sentence aloud:")
    st.info(challenge["sentence_text"])
    st.caption(f"Expires in about {remaining} seconds.")

    control_cols = st.columns([1, 4])
    with control_cols[0]:
        if st.button("Re-record audio", key=f"kyc_rerecord_{attempt_no}"):
            st.session_state[f"kyc_audio_nonce_{attempt_no}"] = (
                st.session_state.get(f"kyc_audio_nonce_{attempt_no}", 0) + 1
            )
            st.rerun()
    with control_cols[1]:
        audio_file = _audio_capture(attempt_no)

    if st.button(
        f"Submit attempt {attempt_no}",
        type="primary",
        disabled=not audio_file,
        key=f"kyc_submit_attempt_{attempt_no}",
    ):
        audio_bytes = audio_file.getvalue()
        if not audio_bytes:
            st.error("No audio recorded. Please record your voice and submit again.")
            return
        mime_type = getattr(audio_file, "type", None) or "audio/wav"
        filename = getattr(audio_file, "name", f"attempt_{attempt_no}.wav")
        with st.spinner("Validating audio recording and extracting voice embedding..."):
            ok, payload = api_client.submit_attempt(
                session_id=session_id,
                attempt_no=attempt_no,
                audio_filename=filename,
                audio_bytes=audio_bytes,
                mime_type=mime_type,
            )

        if not ok:
            st.error(f"Submit error: {payload}")
            st.session_state[challenge_key] = None
            if st.button(f"Reattempt recording", key=f"kyc_error_reattempt_{attempt_no}"):
                st.rerun()
            return
        if payload.get("success"):
            st.session_state[attempt_ok_key] = True
            st.success(payload.get("message", f"Attempt {attempt_no} passed!"))
            st.rerun()
        else:
            st.session_state[attempt_ok_key] = False
            st.session_state[challenge_key] = None
            st.error(payload.get("message", "Attempt failed. Please reattempt with a new sentence."))
            if st.button(f"Reattempt recording", key=f"kyc_fail_reattempt_{attempt_no}"):
                st.rerun()


_state("kyc_attempt_1_ok", False)
_state("kyc_attempt_2_ok", False)
_state("kyc_enrolment_done", False)

header_cols = st.columns([4, 1])
with header_cols[0]:
    st.title("Voice KYC Enrolment")
    st.caption("Two unique sentence challenges prevent reusable enrollment recordings.")
with header_cols[1]:
    if st.button("Back to Dashboard"):
        st.switch_page("app.py")

st.divider()

st.subheader("1. Phone number")
with st.form("kyc_start_form"):
    phone_number = st.text_input(
        "Customer phone number",
        value=st.session_state.get("kyc_phone_number", ""),
        placeholder="+919876543210",
    )
    start_clicked = st.form_submit_button("Start KYC session", type="primary")

if start_clicked:
    ok, payload = api_client.start_kyc_session(phone_number)
    if ok:
        _reset_session()
        st.session_state["kyc_session_id"] = payload["session_id"]
        st.session_state["kyc_phone_number"] = phone_number
        st.success(f"KYC session started: {payload['session_id']}")
    else:
        st.error(payload)

session_id = _state("kyc_session_id")
if not session_id:
    st.info("Start a KYC session to unlock the two voice attempts.")
    st.stop()

st.caption(f"Active session: {session_id}")

st.divider()
st.subheader("2. Attempt 1")
_attempt_panel(1)

st.divider()
st.subheader("3. Attempt 2")
if not st.session_state.get("kyc_attempt_1_ok"):
    st.warning("Attempt 1 must pass before attempt 2.")
else:
    _attempt_panel(2)

st.divider()
st.subheader("4. Enrol")
both_attempts_ok = st.session_state.get("kyc_attempt_1_ok") and st.session_state.get("kyc_attempt_2_ok")
if not both_attempts_ok:
    st.info("The Enrol button is enabled after both sentence attempts pass.")
else:
    if st.button("Enrol", type="primary", disabled=st.session_state.get("kyc_enrolment_done")):
        with st.spinner("Averaging embeddings and generating Bio-Hash..."):
            ok, payload = api_client.enrol_voice(session_id)
        if ok:
            st.session_state["kyc_enrolment_done"] = True
            st.success(payload["message"])
            st.caption(
                "Each attempt required a unique spoken sentence, so a recorded/replayed "
                "voice clip cannot be reused to enrol."
            )
        else:
            st.error(payload)

st.divider()
if st.button("Start over"):
    _reset_session()
    st.rerun()
