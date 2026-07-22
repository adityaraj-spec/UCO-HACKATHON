"""
PhaseGuard Streamlit dashboard.
"""

import streamlit as st

import api_client
from config import (
    API_BASE_URL,
    HARD_FAIL_THRESHOLD,
    LAYER1_FRAUD_THRESHOLD,
    SIMILARITY_THRESHOLD,
    STEP_UP_LOWER_BOUND,
)

st.set_page_config(page_title="PhaseGuard", layout="wide")

st.title("PhaseGuard Voice KYC")
st.caption("Dynamic sentence enrollment, speaker verification, and transaction risk checks")

health_col, flow_col = st.columns([1, 2])

with health_col:
    st.subheader("Backend")
    st.metric("API Base URL", API_BASE_URL)
    if st.button("Check API health", type="primary"):
        ok, payload = api_client.check_health()
        if ok:
            st.success("API is reachable.")
            st.json(payload)
        else:
            st.error(f"API health check failed: {payload}")

with flow_col:
    st.subheader("Current flow")
    st.write("1. Start Voice KYC with the customer's phone number.")
    st.write("2. Read two fresh server-generated sentences, one per attempt.")
    st.write("3. Submit each audio recording before expiry.")
    st.write("4. Enrol only after both attempts pass, then continue to verification/admin approval.")

st.divider()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Pass Threshold", f"{SIMILARITY_THRESHOLD:.4f}")
col2.metric("Step-Up Bound", f"{STEP_UP_LOWER_BOUND:.4f}")
col3.metric("Hard Fail Bound", f"{HARD_FAIL_THRESHOLD:.4f}")
col4.metric("Layer 1 Fraud Limit", f"{LAYER1_FRAUD_THRESHOLD:.2f}")

st.divider()

nav_cols = st.columns(4)
with nav_cols[0]:
    if st.button("Open Voice KYC", use_container_width=True):
        st.switch_page("pages/1_Enrollment.py")
with nav_cols[1]:
    if st.button("Open Verification", use_container_width=True):
        st.switch_page("pages/2_Verification.py")
with nav_cols[2]:
    if st.button("Open History", use_container_width=True):
        st.switch_page("pages/3_History.py")
with nav_cols[3]:
    if st.button("Open Risk Dashboard", use_container_width=True):
        st.switch_page("pages/4_Risk_Dashboard.py")

st.info(
    "Enrollment now rejects reusable recordings by requiring two unique, expiring "
    "sentence challenges before the Bio-Hash is generated."
)
