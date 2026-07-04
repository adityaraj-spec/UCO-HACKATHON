"""Voice enrollment using the authenticated banking session."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

import api_client

st.set_page_config(page_title="PhaseGuard Enrollment", page_icon="EN", layout="wide")
st.title("Voice Enrollment")

user = st.session_state.get("current_user")
if not st.session_state.get("session_token") or not user:
    st.warning("Please log in from the main page first.")
    st.stop()

ok, profile = api_client.get_me()
if ok:
    user.update(profile)

c1, c2, c3 = st.columns(3)
c1.metric("Customer", user.get("full_name", "-"))
c2.metric("Account", user.get("account_number", "-"))
c3.metric("KYC", user.get("kyc_status", "-"))

if user.get("kyc_status") == "REJECTED":
    st.error("KYC is rejected. Voice enrollment is blocked.")
    st.stop()
if user.get("kyc_status") == "PENDING":
    st.warning("KYC is pending. Login is allowed, but production policy blocks voice enrollment until verification.")
    st.stop()

st.divider()

if not st.session_state.get("consent_granted"):
    st.subheader("Biometric Consent")
    if st.button("Grant Consent", type="primary"):
        ok, payload = api_client.grant_consent(None)
        if ok:
            st.session_state["consent_granted"] = True
            st.rerun()
        st.error(payload)
    st.stop()

session = st.session_state.get("enroll_session")
if session is None:
    st.subheader("Enrollment Session")
    if st.button("Start Enrollment Session", type="primary"):
        ok, payload = api_client.create_enrollment_session(None)
        if ok:
            st.session_state["enroll_session"] = payload
            st.rerun()
        st.error(payload)
    st.stop()

samples_submitted = session.get("samples_submitted", 0)
samples_required = session.get("samples_required", 5)
st.progress(samples_submitted / samples_required, text=f"{samples_submitted}/{samples_required} samples uploaded")

if samples_submitted < samples_required:
    remaining = samples_required - samples_submitted
    uploaded = st.file_uploader(
        f"Upload up to {remaining} remaining samples",
        type=["wav", "flac", "mp3", "ogg", "m4a"],
        accept_multiple_files=True,
    )
    if uploaded and st.button("Submit Samples", type="primary"):
        latest_payload = None
        submitted = 0
        for item in uploaded[:remaining]:
            ok, payload = api_client.submit_enrollment_sample(
                user_id=None,
                session_id=session["session_id"],
                filename=item.name,
                file_bytes=item.getvalue(),
                mime_type=item.type or "audio/wav",
            )
            if not ok:
                st.error(payload)
                st.stop()
            latest_payload = payload
            submitted += 1
        if latest_payload:
            st.session_state["enroll_session"]["samples_submitted"] = latest_payload["samples_submitted"]
            st.success(f"{submitted} sample(s) accepted.")
            st.rerun()
    st.stop()

st.success("All required samples are uploaded.")
if st.button("Finalize Enrollment", type="primary"):
    ok, payload = api_client.complete_enrollment(None, session["session_id"])
    if ok:
        st.success(payload.get("message", "Enrollment completed."))
        st.session_state.pop("enroll_session", None)
        st.session_state.pop("consent_granted", None)
    else:
        st.error(payload)
