"""
streamlit_app/pages/1_Enrollment.py

Voice Enrollment page — 5-step session-based enrollment flow aligned to
the PhaseGuard Layer 2 backend:

  Step 1: Create or look up a user by name + email (no raw UUID)
  Step 2: Grant DPDP biometric consent   → POST /consent/grant
  Step 3: Start enrollment session        → POST /voice/enroll/session
  Step 4: Upload 5 audio samples          → POST /voice/enroll/sample  ×5
  Step 5: Finalize                        → POST /voice/enroll/complete
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import api_client

st.set_page_config(page_title="PhaseGuard – Enrollment", page_icon="🎙️", layout="wide")
st.title("🎙️ Voice Enrollment")
st.caption("Register a customer's voiceprint using the 5-sample session-based flow")

# ── helpers ──────────────────────────────────────────────────────────
def current_user() -> dict | None:
    return st.session_state.get("current_user")

def _user_badge() -> None:
    u = current_user()
    if u:
        st.success(f"👤 Active user: **{u['name']}** ({u['email']})")

# ══════════════════════════════════════════════════════════════════════
# STEP 1 — Create or look up user
# ══════════════════════════════════════════════════════════════════════
with st.expander("Step 1 — Create or look up user", expanded=current_user() is None):
    sub_tab_new, sub_tab_existing = st.tabs(["➕ Create new user", "🔍 Look up by email"])

    # -- create new --
    with sub_tab_new:
        with st.form("create_user_form"):
            col1, col2 = st.columns(2)
            name = col1.text_input("Full name", placeholder="e.g. Priya Sharma")
            email = col2.text_input("Email address", placeholder="e.g. priya@example.com")
            submitted = st.form_submit_button("Create user", type="primary")

        if submitted:
            if not name or not email:
                st.warning("Please provide both a name and an email.")
            else:
                ok, payload = api_client.create_user(name, email)
                if ok:
                    st.session_state["current_user"] = payload
                    st.session_state.pop("consent_granted", None)
                    st.session_state.pop("enroll_session", None)
                    st.success(f"✅ User **{payload['name']}** created successfully.")
                    st.rerun()
                else:
                    st.error(f"Failed to create user: {payload}")

    # -- look up existing --
    with sub_tab_existing:
        with st.form("lookup_user_form"):
            lookup_id = st.text_input("User ID (UUID)", placeholder="Paste the UUID from a previous session")
            looked_up = st.form_submit_button("Find user", type="primary")

        if looked_up and lookup_id:
            ok, payload = api_client.get_user(lookup_id)
            if ok:
                st.session_state["current_user"] = payload
                st.session_state.pop("consent_granted", None)
                st.session_state.pop("enroll_session", None)
                st.success(f"✅ Found user: **{payload['name']}** ({payload['email']})")
                st.rerun()
            else:
                st.error(f"User not found: {payload}")

_user_badge()

if not current_user():
    st.stop()

user = current_user()
user_id = user["id"]

st.divider()

# ══════════════════════════════════════════════════════════════════════
# STEP 2 — Grant DPDP Consent
# ══════════════════════════════════════════════════════════════════════
consent_granted = st.session_state.get("consent_granted", False)

with st.expander("Step 2 — Grant DPDP Biometric Consent", expanded=not consent_granted):
    st.info(
        "Under India's **Digital Personal Data Protection (DPDP) Act**, explicit "
        "customer consent is required before storing biometric voice data. "
        "This must be granted **before** starting enrollment."
    )
    if consent_granted:
        st.success("✅ Consent already granted for this session.")
    else:
        if st.button("✅ Grant Consent", type="primary", key="btn_consent"):
            ok, payload = api_client.grant_consent(user_id)
            if ok:
                st.session_state["consent_granted"] = True
                st.success("Consent recorded. You may now start enrollment.")
                st.rerun()
            else:
                st.error(f"Failed to record consent: {payload}")

if not consent_granted:
    st.stop()

st.divider()

# ══════════════════════════════════════════════════════════════════════
# STEP 3 — Start Enrollment Session
# ══════════════════════════════════════════════════════════════════════
enroll_session = st.session_state.get("enroll_session")

with st.expander("Step 3 — Start Enrollment Session", expanded=enroll_session is None):
    if enroll_session:
        s = enroll_session
        st.success(
            f"✅ Session **{s['session_id'][:8]}…** is active — "
            f"{s['samples_submitted']}/{s['samples_required']} samples submitted."
        )
        if st.button("🔄 Restart session (create new)", key="btn_restart"):
            st.session_state.pop("enroll_session", None)
            st.rerun()
    else:
        st.write("Click below to initialise a new 5-sample enrollment session.")
        if st.button("▶ Start Enrollment Session", type="primary", key="btn_start_session"):
            ok, payload = api_client.create_enrollment_session(user_id)
            if ok:
                st.session_state["enroll_session"] = payload
                st.success(f"Session started. Upload {payload['samples_required']} audio samples below.")
                st.rerun()
            else:
                st.error(f"Failed to start session: {payload}")

if not enroll_session:
    st.stop()

st.divider()

# ══════════════════════════════════════════════════════════════════════
# STEP 4 — Submit Audio Samples (one at a time)
# ══════════════════════════════════════════════════════════════════════
session_id = enroll_session["session_id"]
samples_required = enroll_session.get("samples_required", 5)
samples_submitted = enroll_session.get("samples_submitted", 0)

st.subheader(f"Step 4 — Upload Audio Samples ({samples_submitted}/{samples_required})")

# Progress bar
st.progress(samples_submitted / samples_required, text=f"{samples_submitted}/{samples_required} samples uploaded")

if samples_submitted < samples_required:
    st.info(
        f"Upload **one WAV file at a time**. "
        f"You need **{samples_required - samples_submitted}** more sample(s). "
        "Each sample should be 3-10 seconds of natural speech."
    )

    uploaded = st.file_uploader(
        f"Upload sample {samples_submitted + 1} of {samples_required}",
        type=["wav", "flac", "mp3", "ogg", "m4a"],
        key=f"sample_upload_{samples_submitted}",
    )

    if uploaded:
        submit_col, _ = st.columns([1, 3])
        if submit_col.button(
            f"📤 Submit Sample {samples_submitted + 1}",
            type="primary",
            key=f"btn_submit_{samples_submitted}",
        ):
            with st.spinner("Processing audio sample…"):
                ok, payload = api_client.submit_enrollment_sample(
                    user_id=user_id,
                    session_id=session_id,
                    filename=uploaded.name,
                    file_bytes=uploaded.getvalue(),
                    mime_type=uploaded.type or "audio/wav",
                )

            if ok:
                # Update local session state counter
                st.session_state["enroll_session"]["samples_submitted"] = payload["samples_submitted"]

                st.success(f"✅ Sample {payload['samples_submitted']}/{payload['samples_required']} accepted!")

                # Show quality report
                if "quality_report" in payload:
                    qr = payload["quality_report"]
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("SNR", f"{qr['snr_db']:.1f} dB")
                    c2.metric("Voice Ratio", f"{qr['voice_ratio']:.0%}")
                    c3.metric("Quality Score", f"{qr['quality_score']:.2f}")
                    c4.metric("Duration", f"{qr['duration_seconds']:.1f}s")

                if payload.get("is_complete"):
                    st.info("All samples collected! Proceed to Step 5 to finalise enrollment.")
                st.rerun()
            else:
                st.error(f"Sample rejected: {payload}")

else:
    st.success("✅ All required samples have been submitted. Complete enrollment below.")

st.divider()

# ══════════════════════════════════════════════════════════════════════
# STEP 5 — Finalize Enrollment
# ══════════════════════════════════════════════════════════════════════
st.subheader("Step 5 — Finalise Enrollment")

if samples_submitted < samples_required:
    st.warning(
        f"You still need {samples_required - samples_submitted} more sample(s) "
        "before you can finalise."
    )
else:
    st.write(
        "All samples are ready. Click below to compute the BioHash anchor embedding, "
        "encrypt and store the voice template in the vault."
    )
    if st.button("🔐 Finalise Enrollment", type="primary", key="btn_finalise"):
        with st.spinner("Computing anchor embedding and encrypting voiceprint…"):
            ok, payload = api_client.complete_enrollment(user_id, session_id)

        if ok:
            st.balloons()
            st.success(f"🎉 {payload.get('message', 'Enrollment completed!')}")
            col1, col2 = st.columns(2)
            col1.metric("Quality Level", payload.get("quality_level", "—"))
            col2.metric("Status", "Enrolled ✅")
            # Clear session data so a fresh enrollment can start
            st.session_state.pop("enroll_session", None)
            st.session_state.pop("consent_granted", None)
        else:
            st.error(f"Finalisation failed: {payload}")
