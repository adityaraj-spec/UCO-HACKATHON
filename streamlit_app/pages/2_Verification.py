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
from config import (
    HARD_FAIL_THRESHOLD,
    LAYER1_FRAUD_THRESHOLD,
    SIMILARITY_THRESHOLD,
    STEP_UP_LOWER_BOUND,
)

st.set_page_config(page_title="PhaseGuard - Verification", page_icon="🔐", layout="wide")

st.title("🔐 Verification & FAISS Search")
st.caption("Verify a live audio sample or perform 1-to-N speaker search across all templates")

tab1, tab2 = st.tabs(["1-to-1 Verification", "1-to-N FAISS Identification"])

with tab1:
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
        key="verify_file",
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
        key="verify_layer1",
    )

    st.subheader("3. Optional transaction context")
    transaction_enabled = st.checkbox("This verification is for a banking transaction")
    transaction_type = None
    transaction_amount = None
    expected_phrase = None
    spoken_text = None
    if transaction_enabled:
        ctx_cols = st.columns(2)
        transaction_type = ctx_cols[0].selectbox(
            "Transaction type",
            ["balance_inquiry", "fund_transfer", "add_payee", "limit_change"],
        )
        transaction_amount = ctx_cols[1].number_input(
            "Transaction amount",
            min_value=0.0,
            value=0.0,
            step=100.0,
        )
        expected_phrase = st.text_input(
            "Expected dynamic phrase",
            placeholder="transfer 25000 to account 1234",
        )
        spoken_text = st.text_input(
            "ASR transcript",
            placeholder="transfer twenty five thousand to account 1234",
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
                transaction_type=transaction_type,
                transaction_amount=transaction_amount if transaction_enabled else None,
                expected_phrase=expected_phrase,
                spoken_text=spoken_text,
            )

        if not ok:
            st.error(f"Verification failed: {payload}")
        else:
            st.divider()
            st.subheader("Result")

            sim_score = payload["similarity_score"]

            # ---- Identity decision ----
            if payload["verified"]:
                st.success(
                    f"✅ Identity confirmed — similarity "
                    f"{sim_score:.4f} ≥ Pass Threshold ({SIMILARITY_THRESHOLD:.4f})"
                )
            elif sim_score >= STEP_UP_LOWER_BOUND:
                st.warning(
                    f"⚠️ Moderate similarity — similarity "
                    f"{sim_score:.4f} in Step-Up zone ({STEP_UP_LOWER_BOUND:.4f} - {SIMILARITY_THRESHOLD:.4f})"
                )
            else:
                st.error(
                    f"🚫 Hard Mismatch — similarity "
                    f"{sim_score:.4f} < Step-Up Bound ({STEP_UP_LOWER_BOUND:.4f})"
                )

            # ---- Risk level banner ----
            risk_level = payload["risk_level"]
            if risk_level == "FRAUD_ALERT":
                st.error(f"🔴 Risk Level: **{risk_level}**")
            else:
                st.success(f"🟢 Risk Level: **{risk_level}**")

            col1, col2, col3 = st.columns(3)
            col1.metric("Cosine Similarity", f"{sim_score:.4f}")
            col2.metric("Layer 1 Score", f"{payload['layer1_score']:.3f}")
            col3.metric("Risk Score", f"{payload['risk_score']:.1f}%")

            if payload.get("transaction_tier"):
                tcol1, tcol2, tcol3, tcol4 = st.columns(4)
                tcol1.metric("Transaction Tier", payload["transaction_tier"])
                tcol2.metric("Phrase Match", "Yes" if payload.get("phrase_match") else "No")
                tcol3.metric("OTP Required", "Yes" if payload.get("otp_required") else "No")
                tcol4.metric("Final Authorized", "Yes" if payload.get("final_authorized") else "No")

            # ---- Gauges ----
            gauge_col1, gauge_col2 = st.columns(2)

            with gauge_col1:
                fig_sim = go.Figure(
                    go.Indicator(
                        mode="gauge+number",
                        value=sim_score,
                        title={"text": "Speaker Similarity (Layer 2)"},
                        gauge={
                            "axis": {"range": [-1, 1]},
                            "bar": {
                                "color": "#4caf50"
                                if sim_score >= SIMILARITY_THRESHOLD
                                else ("#ff9800" if sim_score >= STEP_UP_LOWER_BOUND else "#f44336")
                            },
                            "threshold": {
                                "line": {"color": "green", "width": 3},
                                "thickness": 0.75,
                                "value": SIMILARITY_THRESHOLD,
                            },
                        },
                    )
                )
                fig_sim.update_layout(height=280, margin=dict(t=50, b=10))
                st.plotly_chart(fig_sim, use_container_width=True)

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
                st.plotly_chart(fig_risk, use_container_width=True)

            st.json(payload)

with tab2:
    st.subheader("1-to-N FAISS Vector Search")
    st.caption("Identify a speaker by comparing audio against all enrolled voiceprints in-memory.")

    identify_file = st.file_uploader(
        "Upload audio recording to identify speaker",
        type=["wav", "flac", "mp3", "ogg", "m4a"],
        key="identify_file",
    )
    top_k = st.slider("Top K matches", min_value=1, max_value=10, value=5)

    if st.button("🔎 Identify Speaker across All Templates", type="primary", disabled=not identify_file):
        with st.spinner("Searching FAISS index..."):
            ok, payload = api_client.identify_speaker(
                filename=identify_file.name,
                file_bytes=identify_file.getvalue(),
                mime_type=identify_file.type or "audio/wav",
                k=top_k,
            )

        if not ok:
            st.error(f"Identification failed: {payload}")
        else:
            matches = payload.get("matches", [])
            if not matches:
                st.info("No matching voiceprints found in FAISS index.")
            else:
                st.success(f"Found {len(matches)} match(es) from FAISS index:")
                for rank, match in enumerate(matches, 1):
                    score = match["similarity_score"]
                    uid = match["user_id"]
                    status_badge = "✅ High Match" if score >= SIMILARITY_THRESHOLD else ("⚠️ Potential Match" if score >= STEP_UP_LOWER_BOUND else "❌ Low Match")
                    st.write(f"**Rank {rank}**: User `{uid}` — Similarity: `{score:.4f}` ({status_badge})")
            st.json(payload)
