"""
streamlit_app/pages/2_Verification.py

Verification page: upload a live audio sample, verify it against an
enrolled user's voiceprint via POST /api/v1/verify, and display the
identity decision plus combined risk assessment.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import plotly.graph_objects as go
import streamlit as st

import api_client
from config import LAYER1_FRAUD_THRESHOLD, SIMILARITY_THRESHOLD

st.set_page_config(page_title="PhaseGuard - Verification", page_icon="🔐", layout="wide")

st.title("🔐 Verification")
st.caption("Verify a live audio sample against an enrolled voiceprint")

# ----------------------------------------------------------------------
# Step 1: Select user
# ----------------------------------------------------------------------
st.subheader("1. Select the claimed user")

default_user_id = st.session_state.get("enroll_user_id", "")
user_id = st.text_input("User ID (UUID)", value=default_user_id, key="verify_user_id")

if user_id:
    ok, payload = api_client.get_user(user_id)
    if ok:
        col1, col2, col3 = st.columns(3)
        col1.metric("Name", payload["name"])
        col2.metric("Enrolled?", "Yes" if payload["is_enrolled"] else "No")
        col3.metric("Recordings on file", payload.get("recording_count", 0))
        if not payload["is_enrolled"]:
            st.warning(
                "This user has not been enrolled yet. "
                "Go to the Enrollment page first."
            )
    else:
        st.error(f"Could not find user: {payload}")

st.divider()

# ----------------------------------------------------------------------
# Step 2: Upload audio + optional Layer 1 score
# ----------------------------------------------------------------------
st.subheader("2. Upload live audio")

uploaded_file = st.file_uploader(
    "Upload WAV/FLAC/MP3 recording to verify",
    type=["wav", "flac", "mp3", "ogg", "m4a"],
)

st.markdown(
    "**Layer 1 score** (optional) — AI-voice-detection probability from "
    "PhaseGuard's Layer 1. If Layer 1 hasn't been run, leave this at 0.0."
)
layer1_score = st.slider(
    "Layer 1 AI-voice probability",
    min_value=0.0,
    max_value=1.0,
    value=0.0,
    step=0.01,
)

verify_clicked = st.button(
    "🔍 Run Verification",
    type="primary",
    disabled=not (user_id and uploaded_file),
)

if verify_clicked:
    with st.spinner("Extracting embedding and computing similarity..."):
        ok, payload = api_client.verify_user(
            user_id=user_id,
            filename=uploaded_file.name,
            file_bytes=uploaded_file.getvalue(),
            mime_type=uploaded_file.type or "audio/wav",
            layer1_score=layer1_score,
        )

    if not ok:
        st.error(f"Verification failed: {payload}")
    else:
        st.divider()
        st.subheader("Result")

        # ---- Identity decision ----
        if payload["verified"]:
            st.success(
                f"✅ Identity confirmed — similarity "
                f"{payload['similarity_score']:.3f} ≥ {SIMILARITY_THRESHOLD:.2f}"
            )
        else:
            st.error(
                f"🚫 Identity mismatch — similarity "
                f"{payload['similarity_score']:.3f} < {SIMILARITY_THRESHOLD:.2f}"
            )

        # ---- Risk level banner ----
        risk_level = payload["risk_level"]
        if risk_level == "FRAUD_ALERT":
            st.error(f"🔴 Risk Level: **{risk_level}**")
        else:
            st.success(f"🟢 Risk Level: **{risk_level}**")

        col1, col2, col3 = st.columns(3)
        col1.metric("Cosine Similarity", f"{payload['similarity_score']:.3f}")
        col2.metric("Layer 1 Score", f"{payload['layer1_score']:.3f}")
        col3.metric("Risk Score", f"{payload['risk_score']:.1f}%")

        # ---- Gauges ----
        gauge_col1, gauge_col2 = st.columns(2)

        with gauge_col1:
            fig_sim = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=payload["similarity_score"],
                    title={"text": "Speaker Similarity (Layer 2)"},
                    gauge={
                        "axis": {"range": [-1, 1]},
                        "bar": {
                            "color": "#4caf50"
                            if payload["similarity_score"] >= SIMILARITY_THRESHOLD
                            else "#f44336"
                        },
                        "threshold": {
                            "line": {"color": "orange", "width": 3},
                            "thickness": 0.75,
                            "value": SIMILARITY_THRESHOLD,
                        },
                    },
                )
            )
            fig_sim.update_layout(height=280, margin=dict(t=50, b=10))
            st.plotly_chart(fig_sim, width='stretch')

        with gauge_col2:
            fig_risk = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=payload["risk_score"],
                    number={"suffix": "%"},
                    title={"text": "Combined Risk Score"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {
                            "color": "#f44336"
                            if risk_level == "FRAUD_ALERT"
                            else "#4caf50"
                        },
                        "steps": [
                            {"range": [0, 50], "color": "#e8f5e9"},
                            {"range": [50, 100], "color": "#ffebee"},
                        ],
                    },
                )
            )
            fig_risk.update_layout(height=280, margin=dict(t=50, b=10))
            st.plotly_chart(fig_risk, width='stretch')

        st.json(payload)
