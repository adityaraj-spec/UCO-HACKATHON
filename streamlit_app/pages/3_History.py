"""
streamlit_app/pages/3_History.py

History page — verification and risk history for the current user.
User is taken from session state; no raw UUID input shown.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import streamlit as st

import api_client

st.set_page_config(page_title="PhaseGuard – History", page_icon="📜", layout="wide")
st.title("📜 Verification & Risk History")
st.caption("Review past verification attempts and risk evaluations")

# ── User resolution ──────────────────────────────────────────────────
user = st.session_state.get("current_user")

if not user:
    st.warning(
        "No active user in this session. "
        "Please go to the **Enrollment** page and create or look up a user first."
    )
    st.stop()

# User header — name + email only, no UUID
col1, col2, col3 = st.columns(3)
col1.metric("Name", user["name"])
col2.metric("Email", user["email"])

ok_u, u_detail = api_client.get_user(user["id"])
if ok_u:
    col3.metric("Enrolled?", "Yes ✅" if u_detail.get("is_enrolled") else "No ❌")
else:
    col3.metric("Enrolled?", "—")

st.divider()

limit = st.slider("Number of records to show", min_value=5, max_value=200, value=25)
user_id = user["id"]

# ══════════════════════════════════════════════════════════════════════
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
            use_container_width=True,
            hide_index=True,
        )

        st.line_chart(df.set_index("created_at")["similarity_score"], height=250)

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
            use_container_width=True,
            hide_index=True,
        )

        st.line_chart(df.set_index("created_at")["risk_score"], height=250)

        fraud_count = (df["risk_level"] == "FRAUD_ALERT").sum()
        clean_count = (df["risk_level"] == "CLEAN").sum()
        c1, c2 = st.columns(2)
        c1.metric("CLEAN events", int(clean_count))
        c2.metric("FRAUD_ALERT events", int(fraud_count))
