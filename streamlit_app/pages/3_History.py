"""
streamlit_app/pages/3_History.py

History page: review past verification attempts and risk evaluations for a
given user via GET /api/v1/verification-history/{id} and
GET /api/v1/risk-history/{id}.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import streamlit as st

import api_client

st.set_page_config(page_title="PhaseGuard - History", page_icon="📜", layout="wide")

st.title("📜 Verification & Risk History")
st.caption("Review past verification attempts and risk evaluations for a user")

default_user_id = st.session_state.get("enroll_user_id", "")
user_id = st.text_input("User ID (UUID)", value=default_user_id, key="history_user_id")

limit = st.slider("Number of records to show", min_value=5, max_value=200, value=25)

if not user_id:
    st.info("Enter a User ID above to view their history.")
    st.stop()

ok_user, user_payload = api_client.get_user(user_id)
if ok_user:
    col1, col2, col3 = st.columns(3)
    col1.metric("Name", user_payload["name"])
    col2.metric("Email", user_payload["email"])
    col3.metric("Enrolled?", "Yes" if user_payload["is_enrolled"] else "No")
else:
    st.error(f"Could not find user: {user_payload}")
    st.stop()

st.divider()

tab_verification, tab_risk = st.tabs(["🔐 Verification History", "⚠️ Risk History"])

with tab_verification:
    ok, payload = api_client.get_verification_history(user_id, limit=limit)
    if not ok:
        st.error(f"Failed to load verification history: {payload}")
    elif not payload:
        st.info("No verification attempts recorded yet for this user.")
    else:
        df = pd.DataFrame(payload)
        df["created_at"] = pd.to_datetime(df["created_at"])
        df = df.sort_values("created_at", ascending=False)

        st.dataframe(
            df[["created_at", "similarity_score", "decision"]],
            width='stretch',
            hide_index=True,
        )

        st.line_chart(
            df.set_index("created_at")["similarity_score"],
            height=250,
        )

with tab_risk:
    ok, payload = api_client.get_risk_history(user_id, limit=limit)
    if not ok:
        st.error(f"Failed to load risk history: {payload}")
    elif not payload:
        st.info("No risk evaluations recorded yet for this user.")
    else:
        df = pd.DataFrame(payload)
        df["created_at"] = pd.to_datetime(df["created_at"])
        df = df.sort_values("created_at", ascending=False)

        st.dataframe(
            df[["created_at", "risk_score", "risk_level"]],
            width='stretch',
            hide_index=True,
        )

        st.line_chart(
            df.set_index("created_at")["risk_score"],
            height=250,
        )

        fraud_count = (df["risk_level"] == "FRAUD_ALERT").sum()
        clean_count = (df["risk_level"] == "CLEAN").sum()
        col1, col2 = st.columns(2)
        col1.metric("CLEAN events", int(clean_count))
        col2.metric("FRAUD_ALERT events", int(fraud_count))
