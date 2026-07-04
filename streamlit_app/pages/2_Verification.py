"""Voice verification for the authenticated banking customer."""

from __future__ import annotations

import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import plotly.graph_objects as go
import streamlit as st

import api_client
from config import SIMILARITY_THRESHOLD

st.set_page_config(page_title="PhaseGuard Verification", page_icon="VF", layout="wide")
st.title("Voice Verification")

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
c3.metric("Voice enrolled", "Yes" if user.get("is_enrolled") else "No")

if not user.get("is_enrolled"):
    st.warning("This account has no active voice enrollment.")
    st.stop()

mode = "Challenge-Response"

def _device_fp() -> str:
    if "_device_fp" not in st.session_state:
        st.session_state["_device_fp"] = str(uuid.uuid4())
    return st.session_state["_device_fp"]


def _render_result(ok: bool, payload) -> None:
    if not ok:
        st.error(payload)
        return
    verified = payload.get("verified", payload.get("authenticated", False))
    sim_score = payload.get("similarity_score", payload.get("cosine_similarity", 0.0))
    risk_level = payload.get("risk_level", "UNKNOWN")
    risk_score = payload.get("risk_score", payload.get("combined_risk_score", 0.0))

    st.success("Identity confirmed." if verified else "Identity rejected.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Similarity", f"{sim_score:.3f}")
    c2.metric("Risk", risk_level)
    c3.metric("Risk Score", f"{risk_score:.1f}%")

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=sim_score,
            title={"text": "Speaker Similarity"},
            gauge={
                "axis": {"range": [-1, 1]},
                "bar": {"color": "#2e7d32" if sim_score >= SIMILARITY_THRESHOLD else "#c62828"},
                "threshold": {"line": {"color": "#f9a825", "width": 3}, "value": SIMILARITY_THRESHOLD},
            },
        )
    )
    fig.update_layout(height=280, margin=dict(t=50, b=10))
    st.plotly_chart(fig, use_container_width=True)


if mode == "Challenge-Response":
    challenge = st.session_state.get("verify_challenge")
    if not challenge:
        if st.button("Get Challenge Phrase", type="primary"):
            session_id = str(uuid.uuid4())
            st.session_state["verify_session_id"] = session_id
            ok, payload = api_client.get_challenge(None, session_id)
            if ok:
                st.session_state["verify_challenge"] = payload
                st.rerun()
            st.error(payload)
        st.stop()

    st.info(challenge["phrase_text"])
    uploaded = st.file_uploader("Upload spoken challenge recording", type=["wav", "flac", "mp3", "ogg", "m4a"])
    if uploaded and st.button("Verify Challenge", type="primary"):
        with st.spinner("Verifying challenge response..."):
            result = api_client.authenticate_voice(
                user_id=None,
                session_id=st.session_state["verify_session_id"],
                challenge_token=challenge["challenge_token"],
                device_fingerprint=_device_fp(),
                filename=uploaded.name,
                file_bytes=uploaded.getvalue(),
                mime_type=uploaded.type or "audio/wav",
            )
        st.session_state.pop("verify_challenge", None)
        _render_result(*result)
