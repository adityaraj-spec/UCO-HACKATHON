"""
streamlit_app/pages/2_Verification.py

Voice Verification page — challenge-response flow aligned to the
PhaseGuard Layer 2 backend:

  Step 1: Select enrolled user (from session state or by ID lookup)
  Step 2: Get challenge phrase   → GET /voice/challenge
  Step 3: Upload spoken audio    → POST /voice/authenticate
"""

import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import plotly.graph_objects as go
import streamlit as st

import api_client
from config import SIMILARITY_THRESHOLD

st.set_page_config(page_title="PhaseGuard – Verification", page_icon="🔐", layout="wide")
st.title("🔐 Voice Verification")
st.caption("Authenticate a user's identity via spoken challenge-response")

# ── helpers ──────────────────────────────────────────────────────────
def current_user() -> dict | None:
    return st.session_state.get("current_user")

# ══════════════════════════════════════════════════════════════════════
# STEP 1 — Select user
# ══════════════════════════════════════════════════════════════════════
with st.expander("Step 1 — Select user to verify", expanded=current_user() is None):
    # Auto-filled from enrollment session
    if current_user():
        u = current_user()
        st.success(f"👤 Active user: **{u['name']}** ({u['email']})")
        if st.button("Switch user", key="btn_switch_user"):
            st.session_state.pop("current_user", None)
            st.session_state.pop("verify_challenge", None)
            st.rerun()
    else:
        with st.form("verify_lookup_form"):
            lookup_id = st.text_input(
                "User ID (UUID)",
                placeholder="Paste the user UUID from the Enrollment page",
            )
            found = st.form_submit_button("Find user", type="primary")

        if found and lookup_id:
            ok, payload = api_client.get_user(lookup_id)
            if ok:
                st.session_state["current_user"] = payload
                st.rerun()
            else:
                st.error(f"User not found: {payload}")

if not current_user():
    st.info("ℹ️ Complete **Enrollment** first, or look up a user above.")
    st.stop()

user = current_user()
user_id = user["id"]

# Show user card
col1, col2, col3 = st.columns(3)
col1.metric("Name", user["name"])
col2.metric("Email", user["email"])
ok_u, u_detail = api_client.get_user(user_id)
enrolled = u_detail.get("is_enrolled", False) if ok_u else False
col3.metric("Enrolled?", "Yes ✅" if enrolled else "No ❌")

if not enrolled:
    st.warning("This user is not yet enrolled. Please complete enrollment first.")
    st.stop()

st.divider()

# ══════════════════════════════════════════════════════════════════════
# STEP 2 — Get challenge phrase
# ══════════════════════════════════════════════════════════════════════
st.subheader("Step 2 — Get Spoken Challenge Phrase")

verify_challenge = st.session_state.get("verify_challenge")

if verify_challenge:
    st.info(
        f"🗣️ **Please read this phrase aloud and record it:**\n\n"
        f"### _{verify_challenge['phrase_text']}_"
    )
    st.caption(f"Phrase ID: {verify_challenge['phrase_id']}")
    if st.button("🔄 Get a different phrase", key="btn_new_challenge"):
        st.session_state.pop("verify_challenge", None)
        st.rerun()
else:
    st.write(
        "Click below to receive a random challenge phrase. "
        "Read it aloud, record yourself, and upload the audio in Step 3."
    )
    if st.button("🎤 Get Challenge Phrase", type="primary", key="btn_get_challenge"):
        # Generate a temporary verify-session ID for this interaction
        v_session_id = str(uuid.uuid4())
        st.session_state["verify_session_id"] = v_session_id
        ok, payload = api_client.get_challenge(user_id, v_session_id)
        if ok:
            st.session_state["verify_challenge"] = payload
            st.rerun()
        else:
            st.error(f"Failed to get challenge: {payload}")

if not verify_challenge:
    st.stop()

st.divider()

# ══════════════════════════════════════════════════════════════════════
# STEP 3 — Upload audio + Authenticate
# ══════════════════════════════════════════════════════════════════════
st.subheader("Step 3 — Upload Recording and Authenticate")
st.write("Upload a WAV recording of yourself reading the phrase shown above.")

uploaded_file = st.file_uploader(
    "Upload your spoken challenge recording",
    type=["wav", "flac", "mp3", "ogg", "m4a"],
    key="verify_audio_upload",
)

if uploaded_file:
    if st.button("🔍 Verify Identity", type="primary", key="btn_authenticate"):
        v_session_id = st.session_state.get("verify_session_id", str(uuid.uuid4()))
        challenge_token = verify_challenge["challenge_token"]
        # Use a stable device fingerprint (browser session id is fine for demos)
        device_fp = st.session_state.get(
            "_device_fp", str(uuid.uuid4())
        )
        st.session_state["_device_fp"] = device_fp

        with st.spinner("Extracting embedding and running identity check…"):
            ok, payload = api_client.authenticate_voice(
                user_id=user_id,
                session_id=v_session_id,
                challenge_token=challenge_token,
                device_fingerprint=device_fp,
                filename=uploaded_file.name,
                file_bytes=uploaded_file.getvalue(),
                mime_type=uploaded_file.type or "audio/wav",
            )

        st.divider()
        st.subheader("Authentication Result")

        if not ok:
            st.error(f"Authentication failed: {payload}")
        else:
            # ── Decision banner ──
            verified = payload.get("verified", payload.get("authenticated", False))
            sim_score = payload.get("similarity_score", payload.get("cosine_similarity", 0.0))
            risk_level = payload.get("risk_level", "UNKNOWN")
            risk_score = payload.get("risk_score", payload.get("combined_risk_score", 0.0))
            layer1_score = payload.get("layer1_score", 0.0)

            if verified:
                st.success(
                    f"✅ Identity confirmed — similarity {sim_score:.3f} ≥ {SIMILARITY_THRESHOLD:.2f}"
                )
            else:
                st.error(
                    f"🚫 Identity rejected — similarity {sim_score:.3f} < {SIMILARITY_THRESHOLD:.2f}"
                )

            # ── Risk level ──
            if risk_level == "FRAUD_ALERT":
                st.error(f"🔴 Risk Level: **{risk_level}**")
            else:
                st.success(f"🟢 Risk Level: **{risk_level}**")

            # ── Metrics row ──
            c1, c2, c3 = st.columns(3)
            c1.metric("Cosine Similarity", f"{sim_score:.3f}")
            c2.metric("Layer 1 Score", f"{layer1_score:.3f}")
            c3.metric("Risk Score", f"{risk_score:.1f}%")

            # ── Gauges ──
            gauge_col1, gauge_col2 = st.columns(2)

            with gauge_col1:
                fig_sim = go.Figure(
                    go.Indicator(
                        mode="gauge+number",
                        value=sim_score,
                        title={"text": "Speaker Similarity"},
                        gauge={
                            "axis": {"range": [-1, 1]},
                            "bar": {
                                "color": "#4caf50" if sim_score >= SIMILARITY_THRESHOLD else "#f44336"
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
                st.plotly_chart(fig_sim, use_container_width=True)

            with gauge_col2:
                fig_risk = go.Figure(
                    go.Indicator(
                        mode="gauge+number",
                        value=risk_score,
                        number={"suffix": "%"},
                        title={"text": "Combined Risk Score"},
                        gauge={
                            "axis": {"range": [0, 100]},
                            "bar": {
                                "color": "#f44336" if risk_level == "FRAUD_ALERT" else "#4caf50"
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

            with st.expander("Full API response"):
                st.json(payload)

            # Clear challenge after use so next verification gets a fresh phrase
            st.session_state.pop("verify_challenge", None)
