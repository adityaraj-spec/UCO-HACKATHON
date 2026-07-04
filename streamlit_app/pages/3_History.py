"""Verification and risk history for the authenticated account."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import streamlit as st

import api_client

st.set_page_config(page_title="PhaseGuard History", page_icon="HS", layout="wide")
st.title("Verification & Risk History")

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

limit = st.slider("Number of records", min_value=5, max_value=200, value=25)
tab_verification, tab_risk = st.tabs(["Verification History", "Risk History"])

with tab_verification:
    ok, payload = api_client.get_verification_history(limit=limit)
    if not ok:
        st.error(payload)
    elif not payload:
        st.info("No verification attempts recorded yet.")
    else:
        df = pd.DataFrame(payload).sort_values("created_at", ascending=False)
        st.dataframe(df[["created_at", "similarity_score", "decision"]], use_container_width=True, hide_index=True)

with tab_risk:
    ok, payload = api_client.get_risk_history(limit=limit)
    if not ok:
        st.error(payload)
    elif not payload:
        st.info("No risk evaluations recorded yet.")
    else:
        df = pd.DataFrame(payload).sort_values("created_at", ascending=False)
        st.dataframe(df[["created_at", "risk_score", "risk_level"]], use_container_width=True, hide_index=True)
